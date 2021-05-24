import os
from os.path import basename
import shutil
import pathlib

import numpy as np
import torch
import torchvision
import tqdm

from image_processing import hist_match_images, center_crop_images
from learning_utils import LabeledImageDataset
from shared_variables import CACHED_DATASETS_FOLDER
from patch_extraction import SessionPatchExtractor, NegativeExtractionMode
from matplotlib import pyplot as plt

import video_session
import PIL
import PIL.Image


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            raise Exception('Failed to delete %s. Reason: %s' % (file_path, e))


def create_cell_and_no_cell_patches(
        video_sessions=None,
        extraction_mode=SessionPatchExtractor.ALL_MODE,

        limit_to_vessel_mask=False,
        patch_size=(21, 21),
        temporal_width=0,

        mixed_channel_patches=False,

        negative_extraction_mode=NegativeExtractionMode.CIRCLE,

        negative_patch_search_radius=21,
        n_negatives_per_positive=1,

        v=False,
        vv=False
):
    assert type(patch_size) == int or type(patch_size) == tuple or (type(patch_size) == np.ndarray)
    if type(patch_size) == int:
        patch_size = patch_size, patch_size

    if mixed_channel_patches:
        assert temporal_width == 0, \
            'mixed channel patches can not work with temporal patches.' \
            'Make sure that either temporal width is greater than 0 or mixed channel patches is True.'

    if temporal_width > 0:
        # for example, for temporal width 1 we have 1 patch for the next frame 1 for the previous and 1 for current (3).
        n_channels = 2 * temporal_width + 1
    elif mixed_channel_patches:
        # first channel is confocal, second is oa790, third is oa850
        n_channels = 3
    else:
        n_channels = 1

    cell_images = np.zeros([0, *patch_size, n_channels], dtype=np.uint8).squeeze()
    non_cell_images = np.zeros_like(cell_images)

    cell_images_marked = np.zeros_like(cell_images)
    non_cell_images_marked = np.zeros_like(cell_images)

    if video_sessions is None:
        video_sessions = video_session.get_video_sessions(marked=True)
    assert all([vs.has_marked_cells for vs in video_sessions]), 'Not all video sessions have marked cells.'

    if v:
        print('Creating cell and no cell images from videos and cell points csvs...')

    patch_extractor = SessionPatchExtractor(
        video_sessions[0],
        extraction_mode=extraction_mode,

        patch_size=patch_size,
        temporal_width=temporal_width,
        limit_to_vessel_mask=limit_to_vessel_mask,

        negative_extraction_mode=negative_extraction_mode,
        negative_extraction_radius=negative_patch_search_radius,
        n_negatives_per_positive=n_negatives_per_positive,
    )

    for i, session in enumerate(tqdm.tqdm(video_sessions)):
        assert session.has_marked_video, 'Something went wrong.' \
                                         ' get_video_sessions() should have ' \
                                         ' returned that have corresponding marked videos.'
        patch_extractor.session = session

        video_file = session.video_file
        marked_video_file = session.marked_video_oa790_file
        csv_cell_coord_files = session.cell_position_csv_files

        if vv:
            print('Unmarked', basename(video_file), '<->')
            print(*[basename(f) for f in csv_cell_coord_files], sep='\n')

        if temporal_width > 0:
            cur_session_cell_images = patch_extractor.temporal_cell_patches_oa790
            cur_session_non_cell_images = patch_extractor.temporal_non_cell_patches_oa790
        elif mixed_channel_patches:
            cur_session_cell_images = patch_extractor.mixed_channel_cell_patches
            cur_session_non_cell_images = patch_extractor.mixed_channel_non_cell_patches
        else:
            cur_session_cell_images = patch_extractor.cell_patches_oa790
            cur_session_non_cell_images = patch_extractor.non_cell_patches_oa790

        cell_images = np.concatenate((cell_images, cur_session_cell_images), axis=0)
        non_cell_images = np.concatenate((non_cell_images, cur_session_non_cell_images), axis=0)

        if i == 0:
            patch_extractor.visualize_patch_extraction()
            plt.show()

        # Marked patches (useful for debugging)
        if vv:
            print('Marked', basename(marked_video_file), '<->')
            print(*[basename(f) for f in csv_cell_coord_files], sep='\n')

        if temporal_width > 0:
            cur_session_marked_cell_images = patch_extractor.temporal_marked_cell_patches_oa790
            cur_session_marked_non_cell_images = patch_extractor.temporal_marked_non_cell_patches_oa790
        elif mixed_channel_patches:
            cur_session_marked_cell_images = patch_extractor.mixed_channel_marked_cell_patches
            cur_session_marked_non_cell_images = patch_extractor.mixed_channel_marked_non_cell_patches
        else:
            cur_session_marked_cell_images = patch_extractor.marked_cell_patches_oa790
            cur_session_marked_non_cell_images = patch_extractor.marked_non_cell_patches_oa790

        cell_images_marked = np.concatenate((cell_images_marked, cur_session_marked_cell_images), axis=0)
        non_cell_images_marked = np.concatenate((non_cell_images_marked, cur_session_marked_non_cell_images), axis=0)
    if v:
        print(f'Created {len(cell_images)} cell patches and {len(non_cell_images)} non cell patches')
    return cell_images, non_cell_images, cell_images_marked, non_cell_images_marked


def train_test_split_images(images, testset_ratio):
    from sklearn.model_selection import train_test_split
    import random

    image_indices = list(np.arange(len(images)))
    random.shuffle(image_indices)

    trainset_size = int(len(images) * (1 - testset_ratio))
    validset_size = len(images) - trainset_size
    assert trainset_size + validset_size == len(images)

    train_indices, valid_indices = train_test_split(image_indices,
                                                    train_size=trainset_size,
                                                    test_size=validset_size)
    train_images = images[train_indices]
    valid_images = images[valid_indices]

    return train_images, valid_images


def create_dataset_from_patches(
        cell_patches,
        non_cell_patches,

        valid_cell_patches=None,
        valid_non_cell_patches=None,

        standardize=True,
        standardize_mean=.5,
        standardize_std=.5,
        to_grayscale=False,

        random_translation_pixels=0,
        random_rotation_degrees=0,
        center_crop_patch_size=21,

        validset_ratio=0.2,
        v=False):
    """ The cell images are labeled as 1 and non cell images are labeled as 0.

    The cell and non cell images are split to training and validation according to the validset ratio.

    If valid_cell_patches, and valid_non_cell_patches are not None, they are appended to the validation set.

    Args:
        cell_patches: Positive samples to create dataset from. Shape N x H x W (x C)
        non_cell_patches: Negative samples to create dataset from. Shape N x H x W (x C)

        validset_ratio: The validation set ration to split cell and non cell images to validation and training set.

        valid_cell_patches: valid_non_cell_patches: If specified, these images are appended to the positive validation set
            If specified they must have be of shape N X H x W (x C) and must have the same H x W (x C) as the rest of
            the images.
        valid_non_cell_patches: If specified, these images are appended to the negative validation set.
            If specified they must have be of shape N X H x W (x C) and must have the same H x W (x C) as the rest of
            the images.

        standardize (bool):  Set to true to standardize dataset to 0.5 mean and variance.
        to_grayscale (bool): Set to true to make patches with more than 2 channels gray.

        random_translation_pixels (int): Data augmentation technique. If > 0 applies a random translation of the number of
            pixel specified to each sample each time the sample is picked.
            If 0 no translation.
            No augmentation on validation set. (Except cropping)
        random_rotation_degrees (tuple or 0):  Data augmentation technique. If not 0 applies
            and rotation=(a, b) then applies a random rotation from a to b degrees.
            No augmentation on validation set. (Expect cropping)
        center_crop_patch_size:
            Only used with data augmentation technique.
            The final patch size to be cropped from the centre of the images.
            The translation and/or rotation is applied to bigger patches and then their center is cropped to avoid empty pixels.
            Occurs both to training and validation patches.

        v (bool): Verbose flag

    """
    if v:
        print('Creating dataset from cell and non cell patches')
        print('-----------------------------------------------')

    # -- positive patches --
    train_cell_patches, valid_cell_patches_from_split = train_test_split_images(cell_patches, validset_ratio)

    if v:
        print(f'{len(train_cell_patches)} training patches')
        print(f'{len(valid_cell_patches_from_split)} validation patches from split')

    if valid_cell_patches is None:
        valid_cell_patches = valid_cell_patches_from_split
    else:
        if v:
            print(f'{len(valid_cell_patches)} validation patches from files')
        valid_cell_patches = np.concatenate((valid_cell_patches, valid_cell_patches_from_split), axis=0)
        if v:
            print(f'{len(valid_cell_patches)} total validation patches')

    if v:
        print(f'Train cell patches {train_cell_patches.shape}. Valid cell patches {valid_cell_patches.shape}')

    # -- negative patches --
    train_non_cell_patches, valid_non_cell_patches_from_split = train_test_split_images(non_cell_patches,
                                                                                        validset_ratio)

    if valid_non_cell_patches is None:
        valid_non_cell_patches = valid_non_cell_patches_from_split
    else:
        valid_non_cell_patches = np.concatenate((valid_non_cell_patches, valid_non_cell_patches_from_split), axis=0)
    if v:
        print(
            f'✔ Train non cell patches {train_non_cell_patches.shape}. Valid cell patches {valid_non_cell_patches.shape}')

    apply_transforms = random_translation_pixels > 0 or random_translation_pixels != 0
    # -- Trainset --

    train_transforms = None
    if apply_transforms:
        initial_patch_size = cell_patches.shape[1]
        translation_ratio = random_translation_pixels / initial_patch_size
        if v:
            print('Applying transformations to training set.\n '
                  ' Rotation degrees:', random_rotation_degrees,
                  ' Translation pixels:', random_translation_pixels,
                  ' Translation ratio:', translation_ratio)
        train_transforms = [
            torchvision.transforms.RandomAffine(degrees=random_rotation_degrees,
                                                translate=(translation_ratio, translation_ratio),
                                                resample=PIL.Image.LINEAR),
            torchvision.transforms.CenterCrop(center_crop_patch_size)
        ]

    trainset = LabeledImageDataset(
        images=np.concatenate((train_cell_patches, train_non_cell_patches), axis=0),
        labels=np.concatenate(
            (np.ones(len(train_cell_patches), dtype=np.int32), np.zeros(len(train_non_cell_patches), dtype=np.int32)),
            axis=0),

        standardize=standardize,
        mean=standardize_mean,
        std=standardize_std,
        to_grayscale=to_grayscale,

        data_augmentation_transforms=train_transforms
    )

    # -- Validset --
    valid_transforms = None
    if apply_transforms:
        if v:
            print(f'Applying center crop to validation set. Center crop', center_crop_patch_size)
        # Just do center crop on the validation images
        valid_transforms = [torchvision.transforms.CenterCrop(center_crop_patch_size)]

    validset = LabeledImageDataset(
        images=np.concatenate((valid_cell_patches, valid_non_cell_patches), axis=0),
        labels=np.concatenate(
            (np.ones(len(valid_cell_patches), dtype=np.int32), np.zeros(len(valid_non_cell_patches), dtype=np.int32)),
            axis=0),

        standardize=standardize,
        to_grayscale=to_grayscale,

        data_augmentation_transforms=valid_transforms,
    )
    return trainset, validset


def get_cell_and_no_cell_patches(
        video_sessions=None,
        patch_size=(21, 21),
        temporal_width=0,

        mixed_channel_patches=False,
        drop_confocal_channel=True,

        validset_ratio=0.2,
        n_negatives_per_positive=1,
        negative_patch_search_radius=21,
        do_hist_match=False,
        standardize_dataset=True,
        dataset_to_grayscale=False,
        apply_data_augmentation_to_dataset=False,
        try_load_from_cache=True,
        v=False,
        vv=False):
    """ Convenience function to get cell and no cell patches and their corresponding marked(for debugging),
        the torch Datasets, and the template image for histogram matching.

        Firstly checks the cache folder with the datasets to see if the dataset with the exact parameters was already
        created and if not then it creates it and saves it in cache.

    Args:
        validset_ratio (float):  A value between 0.001 and 0.99 of the ratio of the validation set to training set.
        negative_patch_search_radius:
            The radius of the window for the negative patches.
        patch_size (int, tuple): The patch size (height, width) or int for square.

        temporal_width (int):
         If > 0 then this is the number of frames before and after the current frame for the patch.
         The returned patches shape will be patch_height x patch_width x (2 * temporal_width + 1) where the central
         channel will be the patch from the original frame, the channel before that will be the patches from the
         same location but from previous frames and the channels after the central will be from the corresponding
         patches after the central.
         **DOES NOT work with mixed_channel_patches.**

        mixed_channel_patches:
          Whether to use the mixed channel technique.
          The patches returned have two channels, the first is the regular oa790 patch and the second
          channel is the patch at the same position at the registered oa850 frame
          **DOES NOT work with temporal patches. (temporal width must be 0)**
        drop_confocal_channel (bool):
            This option is only for mixed channel patches.
            If True then drops the first channel from the mixed channel patches which is the patch from the
            confocal video.

        video_sessions (list[VideoSessions]):
          The list of video sessions to use. If None automatically uses all video sessions available from
          the data folder.

        n_negatives_per_positive (int):  How many non cells per cell patch.
        do_hist_match (bool):  Whether to histogram matching or not.
        standardize_dataset: Standardizes the values of the datasets to have mean and std -.5, (values [-1, 1])
        dataset_to_grayscale: Makes the datasets output to have 1 channel.

        apply_data_augmentation_to_dataset (bool):
            If true then applies random rotation and translation to the datasets.

        try_load_from_cache (bool):
         If true attempts to read  the data from cache to save time.
         If not true then recreates the data and OVERWRITES the old data.

        v (bool):  Verbose description of what is currently happening.
        vv (bool):  Very Verbose description of what is currently happening.

    Returns:
        (Dataset, Dataset, np array NxHxW, np array,      np array,           np.array,             , np.array h x w of template)
        trainset, validset, cell_patches, non_cell_patches, cell_images_marked, non_cell_images_marked, hist_match_template

        Training set and validation set can be used by torch DataLoader.
        Cell images and non cell images are numpy arrays n x patch_height x patch_width and of type uint8.
        Cell images marked and non marked images are the same but from the marked videos for debugging.
        Histogram match template is the template used for histogram matching. If do_hist_match is False then
        None is returned.
    """
    assert 0.001 <= validset_ratio <= 0.99, f'Validation ratio can not be {validset_ratio}'
    assert type(patch_size) == int or type(patch_size) == tuple
    if type(patch_size) == int:
        height, width = patch_size, patch_size
    elif type(patch_size) == tuple:
        height, width = patch_size

    if mixed_channel_patches:
        assert temporal_width == 0, \
            'mixed channel patches can not work with temporal patches.' \
            'Make sure that either temporal width is greater than 0 or mixed channel patches is True.'

    if v:
        print(f'Patch size {(height, width)}')
        print(f'Temporal width {temporal_width}')
        print(f'Mixed channel patches: {mixed_channel_patches}')
        if mixed_channel_patches:
            print(f'Drop confocal channel: {drop_confocal_channel}')
        print(f'do hist match: {do_hist_match}')
        print(f'Negatives per positive: {n_negatives_per_positive}')
        print(f'Standardize dataset: {standardize_dataset}')
        print(f'Dataset to grayscale: {dataset_to_grayscale}')
        print(f'Data augmentation: {apply_data_augmentation_to_dataset}')
        print()
    if vv:
        v = True

    if not do_hist_match:
        hist_match_template = None

    patch_size = (height, width)

    postfix = f'_ps_{patch_size[0]}_tw_{temporal_width}_mc_{str(mixed_channel_patches).lower()}' \
              f'_dc_{str(drop_confocal_channel).lower()}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}' \
              f'_st_{str(standardize_dataset).lower()}_da_{str(apply_data_augmentation_to_dataset).lower()}'

    dataset_folder = os.path.join(
        CACHED_DATASETS_FOLDER,
        f'dataset{postfix}')
    pathlib.Path(dataset_folder).mkdir(parents=True, exist_ok=True)

    trainset_filename = os.path.join(
        dataset_folder,
        f'trainset.pt')
    validset_filename = os.path.join(
        dataset_folder,
        f'validset.pt')

    cell_images_filename = os.path.join(
        dataset_folder,
        f'cells.npy')
    non_cell_images_filename = os.path.join(
        dataset_folder,
        f'non_cells.npy')
    cell_images_marked_filename = os.path.join(
        dataset_folder,
        f'cells_marked.npy')
    non_cell_images_marked_filename = os.path.join(
        dataset_folder,
        f'non_cells_marked.npy')
    template_image_filename = os.path.join(
        dataset_folder,
        f'hist_match_template_image'
    )

    try:
        if not try_load_from_cache:
            if v:
                print('Not checking cache. Overwriting any old data in the cache.')
            clean_folder(dataset_folder)
            # raise exception to go to catch scope
            raise FileNotFoundError

        if v:
            print('Trying to load data from cache')
            print('--------------------------')

        if v:
            print(f"loading training set from:\t'{trainset_filename}'...")
        trainset = torch.load(trainset_filename)
        if v:
            print(f"loading validation set from:\t'{validset_filename}'...")
        validset = torch.load(validset_filename)

        if v:
            print(f"loading cell patches from:\t'{cell_images_filename}'...")
        cell_images = np.load(cell_images_filename)
        if v:
            print(f"loading non cell patches from:\t'{non_cell_images_filename}'...")
        non_cell_images = np.load(non_cell_images_filename)

        if v:
            print(f"loading marked cell patches from:\t'{cell_images_marked_filename}'...")
        cell_images_marked = np.load(cell_images_marked_filename)
        if v:
            print(f"loading marked non cell patches from:\t'{non_cell_images_marked_filename}'")
        non_cell_images_marked = np.load(non_cell_images_marked_filename)

        if do_hist_match:
            if v:
                print(f"loading histogram matching template image (npy array)")
            hist_match_template = np.load(template_image_filename + '.npy')

        if v:
            print('All data found in cache.')
            print()
    except FileNotFoundError:
        print('--------------------------')
        if try_load_from_cache:
            if vv:
                print('Not all data found fom cache.')
        if v:
            print('Creating datasets... (should not take much time)')

        cell_image_creation_patch_size = patch_size[0]
        if apply_data_augmentation_to_dataset:
            # make the patches bigger so that when we crop after the transformation no black pixels appear.
            translation_pixels = 4
            cell_image_creation_patch_size = patch_size[0] + round(patch_size[0] * .5) + translation_pixels

        cell_images, non_cell_images, cell_images_marked, non_cell_images_marked = \
            create_cell_and_no_cell_patches(patch_size=cell_image_creation_patch_size,
                                            temporal_width=temporal_width,
                                            mixed_channel_patches=mixed_channel_patches,
                                            video_sessions=video_sessions,
                                            n_negatives_per_positive=n_negatives_per_positive,
                                            negative_patch_search_radius=negative_patch_search_radius,
                                            v=v, vv=vv)

        if mixed_channel_patches and drop_confocal_channel:
            # We only want the oa790 and oa850 channel as confocal channel doesn't guarantee that the bloodcell
            # will be visible. (2nd and 3rd channel of mixed channel is oa790 and oa850, 1st is confocal)
            cell_images = cell_images[..., 1:]
            non_cell_images = non_cell_images[..., 1:]
            cell_images_marked = cell_images_marked[..., 1:]
            non_cell_images_marked = non_cell_images_marked[..., 1:]

            if v:
                print(f'Dropped confocal channel from cell images.\n '
                      f'Cell image shape: {cell_images.shape}, Non cell images shape: {non_cell_images.shape}')

        hist_match_template = None
        if do_hist_match:
            if v:
                print('Doing histogram matching')

            if vv:
                print('Doing histogram matching on cell images')
            hist_match_template = cell_images[0]
            cell_images = hist_match_images(cell_images, hist_match_template)
            if vv:
                print('Doing histogram matching on non cell images')

            non_cell_images = hist_match_images(cell_images, hist_match_template)

        trainset, validset = create_dataset_from_patches(
            cell_images, non_cell_images, standardize=standardize_dataset, to_grayscale=dataset_to_grayscale,
            validset_ratio=validset_ratio, v=v,

            random_translation_pixels=4,
            random_rotation_degrees=0,
            center_crop_patch_size=patch_size[0],
        )

        if v:
            print()
            print('Saving datasets')
            print('---------------')

        torch.save(trainset, os.path.join(trainset_filename))
        torch.save(validset, os.path.join(validset_filename))
        np.save(cell_images_filename, cell_images)
        np.save(non_cell_images_filename, non_cell_images)
        np.save(cell_images_marked_filename, cell_images_marked)
        np.save(non_cell_images_marked_filename, non_cell_images_marked)
        # np.save(normalisation_data_range_filename, normalisation_data_range)
        if do_hist_match:
            PIL.Image.fromarray(np.uint8(hist_match_template * 255)).save(template_image_filename + '.png')
            np.save(template_image_filename + '.npy', hist_match_template)

        if v:
            print(f"Saved training set as:\t'{trainset_filename}'")
            print(f"Saved validation set as:\t'{validset_filename}'")
            print()
            print(f"Saved cell images as:\t'{cell_images_filename}'")
            print(f"Saved non cell images as:\t'{non_cell_images_filename}'")
            print()
            print(f"Saved marked cell images (for debugging) as:\t'{cell_images_marked_filename}'")
            print(f"Saved marked non cell images (for debugging) as:\t'{non_cell_images_marked_filename}'")
            if do_hist_match:
                print(f"Saved histogram matching template as:\t {template_image_filename}.png")
                print(f"Saved histogram matching template (npy array) as:\t{template_image_filename}.npy")
            # print(f"Saved normalisation data range as: {normalisation_data_range}.npy")
            print("Cell images array shape:", cell_images.shape)
            print("Non cell images array shape:", non_cell_images.shape)

    if apply_data_augmentation_to_dataset:
        if v:
            print('Doing center crop on patches to bring to original patch size...')
        # bring back patches to their original size
        cell_images = center_crop_images(cell_images, patch_size)
        non_cell_images = center_crop_images(non_cell_images, patch_size)

        cell_images_marked = center_crop_images(cell_images_marked, patch_size)
        non_cell_images_marked = center_crop_images(non_cell_images_marked, patch_size)

    return trainset, validset, \
           cell_images, non_cell_images, \
           cell_images_marked, non_cell_images_marked, \
           hist_match_template


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch-size', default=21, type=int, help='Patch size')
    parser.add_argument('-t', '--temporal-width', default=0, type=int, help='Temporal width of the patches.')
    parser.add_argument('--hist-match', action='store_true',
                        help='Set this flag to do histogram match.')
    parser.add_argument('-n', '--n-negatives-per-positive', default=3, type=int)
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Set to overwrite existing datasets in cache')
    parser.add_argument('-v', default=False, action='store_true',
                        help='Verbose output.')
    parser.add_argument('-vv', default=False, action='store_true',
                        help='Very verbose output.')
    args = parser.parse_args()

    patch_size = args.patch_size, args.patch_size
    hist_match = args.hist_match
    npp = args.n_negatives_per_positive
    overwrite = args.overwrite
    temporal_width = args.temporal_width
    v = args.v
    vv = args.vv

    print('---------------------------------------')
    print('Patch size:', patch_size)
    print('Temporal width:', temporal_width)
    print('hist match:', hist_match)
    print('Negatives per positive:', npp)
    print('---------------------------------------')

    patch_size = 21
    hist_match = False
    nnp = 1
    standardize_dataset = False

    get_cell_and_no_cell_patches(patch_size=patch_size,
                                 n_negatives_per_positive=npp,
                                 do_hist_match=hist_match,
                                 try_load_from_cache=not overwrite,
                                 standardize_dataset=standardize_dataset,
                                 v=v,
                                 vv=vv)


def main_tmp():
    # Input
    patch_size = 21
    do_hist_match = False
    n_negatives_per_positive = 1
    standardize_dataset = True
    temporal_width = 1

    try_load_from_cache = False
    verbose = False
    very_verbose = True

    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    trainset, validset, \
    cell_images, non_cell_images, \
    cell_images_marked, non_cell_images_marked, hist_match_template = \
        get_cell_and_no_cell_patches(patch_size=patch_size,
                                     n_negatives_per_positive=n_negatives_per_positive,
                                     do_hist_match=do_hist_match,
                                     try_load_from_cache=try_load_from_cache,
                                     temporal_width=temporal_width,
                                     standardize_dataset=standardize_dataset,
                                     v=verbose,
                                     vv=very_verbose)


if __name__ == '__main__':
    main_tmp()
#   main()
