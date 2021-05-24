import pathlib
import argparse

import torch
import collections
# import cPickle as pickle
import pickle
import os
from copy import deepcopy

import numpy as np
from cnnlearning import CNN, train, TrainingTracker
from generate_datasets import get_cell_and_no_cell_patches
from classificationutils import classify_labeled_dataset
from shared_variables import CACHED_MODELS_FOLDER


def extract_value_from_string(string, value_prefix, delimiter='_'):
    strings = pathlib.Path(string).name.split(delimiter)
    val = None
    for i, s in enumerate(strings):
        if s == value_prefix:
            # radius = float(re.findall(r"[-+]?\d*\.\d+|\d+", strings[i + 1])[0])
            val = strings[i + 1]
            break

    return val


def load_model_from_cache(
        model,
        train_model=None,
        patch_size=(21, 21),
        n_negatives_per_positive=1,
        mixed_channel=False,
        drop_confocal_channel=False,
        temporal_width=0,
        standardize_dataset=True,
        data_augmentation=False,
        hist_match=False):
    """ Attempts to find the model weights to model from cache.

    Args:
        model: The model to attempt to load.
        patch_size: The patch size used to train the model.
        n_negatives_per_positive:  The number of negatives per positive used to train the model.
        hist_match: Whether histogram matching was used to train the model.

    Returns:
        The filename of the model

    """
    potential_model_directories = [
        f for f in os.listdir(CACHED_MODELS_FOLDER) if os.path.isdir(os.path.join(CACHED_MODELS_FOLDER, f))
    ]
    potential_model_directories = [
        f for f in potential_model_directories if extract_value_from_string(f, 'npp') == str(n_negatives_per_positive)
    ]
    potential_model_directories = [
        f for f in potential_model_directories if extract_value_from_string(f, 'ps') == str(patch_size[0])
    ]
    potential_model_directories = [
        f for f in potential_model_directories if extract_value_from_string(f, 'hm') == str(hist_match).lower()
    ]
    potential_model_directories = [
        f for f in potential_model_directories if extract_value_from_string(f, 'mc') == str(mixed_channel).lower()
    ]
    potential_model_directories = [
        f for f in potential_model_directories if extract_value_from_string(f, 'dc') == str(drop_confocal_channel).lower()
    ]
    potential_model_directories = [
        f for f in potential_model_directories if extract_value_from_string(f, 'st') == str(standardize_dataset).lower()
    ]
    potential_model_directories = [
        f for f in potential_model_directories if extract_value_from_string(f, 'da') == str(data_augmentation).lower()
    ]
    potential_model_directories = [
        f for f in potential_model_directories if extract_value_from_string(f, 'tw') == str(temporal_width)
    ]

    if len(potential_model_directories) == 0:
        raise FileNotFoundError('No model directory in cache')

    best_model_idx = np.argmax([float(extract_value_from_string(f, 'va')) for f in potential_model_directories])
    best_model_directory = os.path.join(CACHED_MODELS_FOLDER, potential_model_directories[best_model_idx])

    best_model_file = \
        [os.path.join(best_model_directory, f) for f in os.listdir(best_model_directory) if f.endswith('.pt')
         and 'train' not in f][0]
    model.load_state_dict(torch.load(best_model_file))
    model.eval()

    if train_model is not None:
        best_model_file = \
            [os.path.join(best_model_directory, f) for f in os.listdir(best_model_directory) if f.endswith('.pt')
             and 'train' in f][0]
        train_model.load_state_dict(torch.load(best_model_file))
        train_model.eval()

    return best_model_directory


# noinspection DuplicatedCode
def train_model_demo(
        video_sessions=None,
        patch_size=(21, 21),
        temporal_width=0,

        mixed_channel_patches=False,
        drop_confocal_channel=True,

        do_hist_match=False,

        standardize_dataset=True,
        dataset_to_grayscale=False,
        apply_data_augmentation_to_dataset=False,
        valid_ratio=0.2,

        n_negatives_per_positive=1,
        train_params=None,
        try_load_data_from_cache=True,
        try_load_model_from_cache=True,
        additional_displays=None,
        device='cuda',
        v=False,
        vv=False,
):
    assert type(patch_size) is int or type(patch_size) is tuple
    if type(patch_size) is int:
        patch_size = patch_size, patch_size

    print('Creating (or loading from cache) data...')
    trainset, validset, \
    cell_images, non_cell_images, \
    cell_images_marked, non_cell_images_marked, \
    hist_match_template = \
        get_cell_and_no_cell_patches(
            video_sessions=video_sessions,
            patch_size=patch_size,
            temporal_width=temporal_width,

            mixed_channel_patches=mixed_channel_patches,
            drop_confocal_channel=drop_confocal_channel,

            do_hist_match=do_hist_match,

            standardize_dataset=standardize_dataset,
            dataset_to_grayscale=dataset_to_grayscale,
            apply_data_augmentation_to_dataset=apply_data_augmentation_to_dataset,
            validset_ratio=valid_ratio,

            n_negatives_per_positive=n_negatives_per_positive,

            try_load_from_cache=try_load_data_from_cache,
            v=v, vv=vv,
        )
    assert len(cell_images.shape) == 3 or len(cell_images.shape) == 4
    assert len(non_cell_images.shape) == 3 or len(non_cell_images.shape) == 4
    assert cell_images.dtype == np.uint8 and non_cell_images.dtype == np.uint8, \
        f'Cell images dtype {cell_images.dtype} non cell images dtype {non_cell_images.dtype}'
    assert cell_images.min() >= 0 and cell_images.max() <= 255
    assert non_cell_images.min() >= 0 and non_cell_images.max() <= 255

    # noinspection PyUnresolvedReferences
    model = CNN(dataset_sample=trainset, output_classes=2).to(device)

    pathlib.Path(CACHED_MODELS_FOLDER).mkdir(parents=True, exist_ok=True)

    if additional_displays is None:
        additional_displays = []
    additional_displays.append(f'Cell patches: {cell_images.shape}. Non cell patches: {non_cell_images.shape}')
    try:
        if not try_load_model_from_cache:
            raise FileNotFoundError

        additional_displays.append(f'Attempting to load model and results from cache with center_crop_patch_size:{patch_size}, '
                                   f' histogram_match: {do_hist_match}, n negatives per positive: {n_negatives_per_positive}')
        model_directory = \
            load_model_from_cache(
                model,
                patch_size=patch_size,
                n_negatives_per_positive=n_negatives_per_positive,
                temporal_width=temporal_width,
                standardize_dataset=standardize_dataset,
                data_augmentation=apply_data_augmentation_to_dataset,
                hist_match=do_hist_match
            )
        print(f"Model found. Loaded model from '{model_directory}'")
        with open(os.path.join(model_directory, 'results.pkl'), 'rb') as results_file:
            results = pickle.load(results_file)

    except FileNotFoundError:
        if try_load_model_from_cache:
            additional_displays.append('Model or results not found in cache.')

        additional_displays.append('Training new model.')
        additional_displays.append('You can interrupt(ctr - C or interrupt kernel in notebook) any time to get '
                                   'the model with the best validation accuracy at the current time.')

        if train_params is None:
            train_params = collections.OrderedDict(
                optimizer=torch.optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4),
                epochs=4000,
                batch_size=1024 * 12,
                do_early_stop=True,
                early_stop_patience=80,
                learning_rate_scheduler_patience=100,
                shuffle=True,

                trainset=trainset,
                validset=validset,
            )
        else:
            train_params = deepcopy(train_params)
            if 'trainset' not in train_params:
                train_params['trainset'] = trainset
            if 'validset' not in train_params:
                train_params['validset'] = validset

        run_configuration_display = collections.OrderedDict({
            'patch size': patch_size[0],
            'temporal width': temporal_width,
            'hist match': do_hist_match,
            'standardize data': standardize_dataset,
            'nnp': n_negatives_per_positive
        })
        additional_displays.append(run_configuration_display)

        if v:
            print('Starting training')
        # set the model in training mode
        model.train()
        # We want to give a little bias to the positive samples because of the unbalanced dataset.
        # The positive samples are 1 / n_negatives_per_positive of the negative samples.
        # We add a 0.1 just to make it a little less biased.
        positive_samples_ratio = min((1 / n_negatives_per_positive) + 0.1, 1.0)
        results: TrainingTracker = train(model, train_params,
                                         criterion=torch.nn.CrossEntropyLoss(torch.tensor([positive_samples_ratio, 1.0]).cuda()),
                                         additional_displays=additional_displays, device=device)

        postfix = f'_ps_{patch_size[0]}_tw_{temporal_width}_mc_{str(mixed_channel_patches).lower()}'\
                  f'_dc_{str(drop_confocal_channel).lower()}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}' \
                  f'_st_{str(standardize_dataset).lower()}_da_{str(apply_data_augmentation_to_dataset).lower()}'

        output_directory = os.path.join(CACHED_MODELS_FOLDER,
                                        f'model_va_{results.recorded_model_valid_accuracy:.3f}{postfix}')
        pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

        print(f'Saving results to directory: {output_directory}')
        results.save(output_directory)
        print('Done')

    model = results.recorded_model.eval()

    _, train_accuracy, train_positive_accuracy, train_negative_accuracy = classify_labeled_dataset(trainset, model, ret_pos_and_neg_acc=True)
    _, valid_accuracy, valid_positive_accuracy, valid_negative_accuracy = classify_labeled_dataset(validset, model, ret_pos_and_neg_acc=True)

    print()
    print(f'Model trained on {len(cell_images)} cell patches and {len(non_cell_images)} non cell patches.')
    print()
    print('Brief evaluation - Best validation accuracy model')
    print('----------------')
    print(f'Epoch:\t', results.recorded_model_epoch)
    print('Training accuracy:\t', f'{train_accuracy:.3f}')
    print('Validation accuracy:\t', f'{valid_accuracy:.3f}')
    print()
    print('Positive train accuracy:\t', f'{train_positive_accuracy:.3f}')
    print('Negative train accuracy:\t', f'{train_negative_accuracy:.3f}')
    print()
    print('Positive valid accuracy:\t', f'{valid_positive_accuracy:.3f}')
    print('Negative valid accuracy:\t', f'{valid_negative_accuracy:.3f}')
    print()

    print(f'{train_accuracy:.3f} {valid_accuracy:.3f} '
          f'{train_positive_accuracy:.3f} {train_negative_accuracy:.3f} '
          f'{valid_positive_accuracy:.3f} {valid_negative_accuracy:.3f}')

    epoch = results.recorded_model_epoch
    print(f'{results.train_accuracies[epoch]:.3f} {results.valid_accuracies[epoch]:.3f} '
          f'{results.train_positive_accuracies[epoch]:.3f} {results.train_negative_accuracies[epoch]:.3f} '
          f'{results.valid_positive_accuracies[epoch]:.3f} {results.valid_negative_accuracies[epoch]:.3f}')

    model = results.recorded_train_model.eval()
    _, train_accuracy, train_positive_accuracy, train_negative_accuracy = classify_labeled_dataset(trainset, model, ret_pos_and_neg_acc=True)
    _, valid_accuracy, valid_positive_accuracy, valid_negative_accuracy = classify_labeled_dataset(validset, model, ret_pos_and_neg_acc=True)

    print()
    print(f'Model trained on {len(cell_images)} cell patches and {len(non_cell_images)} non cell patches.')
    print()
    print('Brief evaluation - Best training accuracy model')
    print('----------------')
    print(f'Epoch:\t', results.recorded_model_epoch)
    print('Training accuracy:\t', f'{train_accuracy:.3f}')
    print('Validation accuracy:\t', f'{valid_accuracy:.3f}')
    print()
    print('Positive train accuracy:\t', f'{train_positive_accuracy:.3f}')
    print('Negative train accuracy:\t', f'{train_negative_accuracy:.3f}')
    print()
    print('Positive valid accuracy:\t', f'{valid_positive_accuracy:.3f}')
    print('Negative valid accuracy:\t', f'{valid_negative_accuracy:.3f}')
    print()

    print(f'{train_accuracy:.3f} {valid_accuracy:.3f} {train_positive_accuracy:.3f} {train_negative_accuracy:.3f} '
          f'{valid_positive_accuracy:.3f} {valid_negative_accuracy:.3f}')

    epoch = results.recorded_train_model_epoch
    print(f'{results.train_accuracies[epoch]:.3f} {results.valid_accuracies[epoch]:.3f} '
          f'{results.train_positive_accuracies[epoch]:.3f} {results.train_negative_accuracies[epoch]:.3f} '
          f'{results.valid_positive_accuracies[epoch]:.3f} {results.valid_negative_accuracies[epoch]:.3f}')

    return model, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ps', '-p', '--patch-size', default=21, type=int, help='Patch size')
    parser.add_argument('-tw', '-t', '--temporal-width', default=0, type=int, help='Temporal width')
    parser.add_argument('-st', '-s', '--standardize', action='store_true',
                        help='Set this flag to standardize dataset output between -1 and 1')
    parser.add_argument('-hm', '--hist-match', action='store_true',
                        help='Set this flag to do histogram match.')
    parser.add_argument('-npp', '-n', '--n-negatives-per-positive', default=3, type=int)
    parser.add_argument('-d', '--device', default='cuda', type=str, help="Device to use for training. 'cuda' or 'cpu'")

    args = parser.parse_args()
    available_devices = ['cuda', 'gpu']
    assert args.device in available_devices, f'Device must be one of {available_devices}.'
    print('---------------------------------------')
    device = args.device
    patch_size = args.patch_size, args.patch_size
    hist_match = args.hist_match
    npp = args.n_negatives_per_positive
    standardize = args.standardize
    temporal_width = args.temporal_width
    print('---------------------------------------')

    train_model_demo(
        patch_size=patch_size,
        do_hist_match=hist_match,
        n_negatives_per_positive=npp,
        standardize_dataset=standardize,
        temporal_width=temporal_width,
        device=device,
        try_load_data_from_cache=True,
        try_load_model_from_cache=True,
    )


if __name__ == '__main__':
    import sys
    from shared_variables import get_video_sessions

    video_sessions = get_video_sessions(registered=True, marked=True)

    train_params = collections.OrderedDict(
        epochs=250,
        lr=.001,

        weight_decay=0.01,
        batch_size=512,  # can be a number or None/'all' to train all trainset at once.
        do_early_stop=True,  # Optional default True
        early_stop_patience=60,  # How many epochs with no validation accuracy improvement before stopping early
        learning_rate_scheduler_patience=20,
        # How many epochs with no validation accuracy improvement before lowering learning rate
        evaluate_epochs=10,
        shuffle=True)

    model, results = train_model_demo(

        patch_size=27,
        temporal_width=0,
        mixed_channel_patches=True,

        do_hist_match=False,
        n_negatives_per_positive=1,

        standardize_dataset=True,
        apply_data_augmentation_to_dataset=False,

        video_sessions=video_sessions[:1],  # The video sessions the data will be created from
        try_load_data_from_cache=True,   # If true attemps to load data from cache. If false just created new (and overwrites old if exist)
        try_load_model_from_cache=True,  # Attemps to load model from cache. If false creates new

        train_params=train_params,  # The training train parameters
        additional_displays=None,
        v=True, vv=True,
    )
    if len(sys.argv) > 1:
        main()
    else:
        main_tmp()
