from typing import Tuple, Dict, Any
import os
import pathlib
import pickle

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation import evaluate_results, EvaluationResults
from image_processing import imextendedmax
from matplotlib import pyplot as plt

import torch
from learning_utils import ImageDataset

from patch_extraction import SessionPatchExtractor, NegativeExtractionMode
from video_session import VideoSession

from enum import Enum, unique

NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1


@unique
class RegionCoordSelectMode(Enum):
    GEOMETRIC_CENTROID = 0
    WEIGHTED_CENTROID = 1
    MAX_INTENSITY_PIXEL = 2


class ClassificationResults:
    def __init__(self,
                 positive_accuracy, negative_accuracy, accuracy, n_positive, n_negative, balanced_accuracy=None,
                 loss=None, ground_truth=None, predictions=None, output_probabilities=None, dataset=None, model=None):
        self.model = model
        self.dataset = dataset

        self.loss = loss

        self.ground_truth = ground_truth
        self.predictions = predictions
        self.output_probabilities = output_probabilities

        self.accuracy = accuracy
        self.balanced_accuracy = balanced_accuracy

        self.positive_accuracy = positive_accuracy
        self.negative_accuracy = negative_accuracy

        self.n_positive = n_positive
        self.n_negative = n_negative

    def __repr__(self):
        import pandas as pd
        pd.DataFrame()
        data = {
            'Balanced Accuracy': self.balanced_accuracy,
            'Accuracy': self.accuracy,
            'Sensitivity': self.positive_accuracy,
            'Specificity': self.negative_accuracy
        }
        df = pd.DataFrame(data, columns=list(data.keys()), index=[0])

        return df.__repr__()


@torch.no_grad()
def classify_labeled_dataset(dataset, model, device="cuda"):
    from sklearn.metrics import balanced_accuracy_score

    model = model.eval()
    model = model.to(device)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
    )

    c = 0
    ground_truth = torch.empty(len(dataset), dtype=torch.long).to(device)
    predictions = torch.empty(len(dataset), dtype=torch.long).to(device)
    output_probabilities = torch.empty((len(dataset), 2), dtype=torch.float32).to(device)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        ground_truth[c:c + len(labels)] = labels

        pred = model(images)
        pred = torch.nn.functional.softmax(pred, dim=1)
        output_probabilities[c:c + len(pred)] = pred

        pred = torch.argmax(pred, dim=1)
        predictions[c:c + len(pred)] = pred

        c += len(pred)

    n_positive_samples = (ground_truth == 1).sum().item()
    n_negative_samples = (ground_truth == 0).sum().item()
    assert n_positive_samples + n_negative_samples == len(dataset)

    n_correct = (ground_truth == predictions).sum().item()
    n_positive_correct = (ground_truth[ground_truth == 1] == predictions[ground_truth == 1]).sum().item()
    n_negative_correct = (ground_truth[ground_truth == 0] == predictions[ground_truth == 0]).sum().item()
    assert n_correct == n_positive_correct + n_negative_correct

    accuracy = n_correct / len(dataset)
    positive_accuracy = n_positive_correct / n_positive_samples
    negative_accuracy = n_negative_correct / n_negative_samples
    balanced_accuracy = (positive_accuracy + negative_accuracy) / 2
    assert np.isclose(balanced_accuracy_score(ground_truth.cpu(), predictions.cpu()), balanced_accuracy)

    return ClassificationResults(
        model=model,
        dataset=dataset,

        ground_truth=ground_truth,
        predictions=predictions,
        output_probabilities=output_probabilities,

        n_positive=n_positive_samples,
        n_negative=n_negative_samples,

        accuracy=accuracy,
        positive_accuracy=positive_accuracy,
        negative_accuracy=negative_accuracy,
        balanced_accuracy=balanced_accuracy
    )


def estimate_cell_positions_from_probability_map(
        probability_map,
        extended_maxima_h,
        region_coord_select_mode=RegionCoordSelectMode.MAX_INTENSITY_PIXEL,
        region_max_threshold=.75,
        sigma=1,
        visualise_intermediate_results=False,
        s=215,
        name='tmp'
):
    assert 0.1 <= extended_maxima_h <= 0.9, f'Extended maxima h must be between .1 and .9 not {extended_maxima_h}'
    from skimage.filters import gaussian
    from skimage import measure

    pm_blurred = gaussian(probability_map, sigma)
    pm_extended_max_bw = imextendedmax(pm_blurred, extended_maxima_h)

    labeled_img, nr_objects = measure.label(pm_extended_max_bw, return_num=True)

    # print(np.where(pm_extended_max_bw)[0])
    pm_extended_max = probability_map.copy()
    pm_extended_max[pm_extended_max_bw] = 0

    # print(pm_extended_max)
    # Notice, the points from the csv is x,y. The result from the probability is y,x so we swap.
    region_props = measure.regionprops(labeled_img, intensity_image=pm_blurred)
    estimated_cell_positions = np.empty((len(region_props), 2))
    i = 0
    culled_regions = []
    for region_idx, region in enumerate(region_props):
        if region.max_intensity <= region_max_threshold:
            culled_regions.append(region_idx)
            continue

        if region_coord_select_mode is RegionCoordSelectMode.MAX_INTENSITY_PIXEL:
            max_intensity_idx = np.argmax(pm_blurred[region.coords[:, 0], region.coords[:, 1]])
            y, x = region.coords[max_intensity_idx]
        elif region_coord_select_mode is RegionCoordSelectMode.WEIGHTED_CENTROID:
            y, x = region.weighted_centroid
        elif region_coord_select_mode is RegionCoordSelectMode.GEOMETRIC_CENTROID:
            y, x = region.centroid

        estimated_cell_positions[i] = x, y
        i += 1
    estimated_cell_positions = estimated_cell_positions[:i]

    if visualise_intermediate_results:
        from plotutils import no_ticks, savefig_tight

        figsize = (50, 35)
        fontsize = 65
        plt.figure(figsize=figsize)
        plt.imshow(probability_map, cmap='jet')
        plt.title('Unprocessed probability map', fontsize=fontsize)
        savefig_tight(f'{name}_1.png')

        plt.figure(figsize=figsize)
        plt.imshow(pm_blurred, cmap='jet')
        plt.title(f'Gaussian Blur with sigma={sigma}', fontsize=fontsize)
        savefig_tight(f'{name}_2.png')

        plt.figure(figsize=figsize)
        plt.imshow(pm_extended_max_bw)
        plt.title(f'Extended maxima transform, H={extended_maxima_h}', fontsize=fontsize)
        savefig_tight(f'{name}_3.png')

        plt.figure(figsize=figsize)
        plt.imshow(pm_extended_max_bw * pm_blurred, cmap='jet')
        plt.title(f'Culling regions with max intensity <= {region_max_threshold}', fontsize=fontsize)
        ax = plt.gca()
        for region_idx in culled_regions:
            region = region_props[region_idx]
            minr, minc, maxr, maxc = region.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, '-r', linewidth=7.5)
            pm_extended_max_bw[region.coords[:, 0], region.coords[:, 1]] = 0
        savefig_tight(f'{name}_4.png')

        plt.figure(figsize=figsize)
        plt.imshow(pm_extended_max_bw)
        plt.scatter(estimated_cell_positions[:, 0], estimated_cell_positions[:, 1], s=s, edgecolors='b',
                    label='estimated locations')
        plt.title(f'Estimated locations. {str(region_coord_select_mode)}', fontsize=fontsize)
        plt.legend(prop={'size': int(fontsize * .65)})
        savefig_tight(f'{name}_5.png')

    return estimated_cell_positions[1:, ...].astype(np.int32)


@torch.no_grad()
def get_label_probability(images, model, standardize=True, to_grayscale=False, n_output_classes=2, device='cuda'):
    """ Make a prediction for the images giving output_probabilities for each labels.

    Arguments:
        images -- NxHxWxC or NxHxW. The images
        model  -- The model to do the prediction

    Returns:
        Returns the probability per label for each image.
    """
    model = model.to(device)
    model = model.eval()

    if len(images.shape) == 3:
        # Add channel dimension when images are single channel grayscale
        # i.e (Nx100x123 -> Nx100x123x1)
        images = images[..., None]

    image_dataset = ImageDataset(images, standardize=standardize, to_grayscale=to_grayscale)
    loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=1024 * 3,
    )

    c = 0
    predictions = torch.empty((len(image_dataset), n_output_classes), dtype=torch.float32)
    for images in loader:
        images = images.to(device)
        pred = model(images)
        pred = torch.nn.functional.softmax(pred, dim=1)
        predictions[c:c + len(pred), ...] = pred
        c += pred.shape[0]

    return predictions


def create_probability_map(patches, model, im_shape, mask=None, standardize=True, to_grayscale=False,
                           device='cuda'):
    if mask is None:
        mask = np.ones(im_shape, dtype=np.bool8)

    mask_indices = np.where(mask.flatten())[0]
    assert len(mask_indices) == len(patches), 'Number of patches must match the number of pixles in masks'

    model = model.to(device)
    model = model.eval()

    label_probabilities = get_label_probability(patches, model, standardize=standardize,
                                                to_grayscale=to_grayscale, device=device)

    probability_map = np.zeros(im_shape, dtype=np.float32)
    rows, cols = np.where(mask)
    probability_map[rows, cols] = label_probabilities[:, 1]

    return probability_map


@torch.no_grad()
def classify_images(images, model, standardize_dataset=True, device="cuda"):
    """ Classify images.

    Arguments:
        images -- NxHxWxC or NxHxW. The images
        model  -- The model to do the prediction

    Returns:
        N predictions. A prediction (label) for each image.
    """
    if len(images.shape) == 3:
        # Add channel dimension when image is single channel grayscale
        # i.e (Nx100x123 -> Nx100x123x1)
        images = images[..., None]

    image_dataset = ImageDataset(images, standardize=standardize_dataset)
    loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=1024 * 5,
        shuffle=False
    )

    c = 0
    predictions = torch.zeros(len(image_dataset), dtype=torch.uint8)
    for batch in loader:
        pred = model(batch.to(device))
        pred = torch.nn.functional.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        predictions[c:c + len(pred)] = pred

        c += len(pred)

    return predictions


class MutualExclusiveArgumentsException(Exception):
    pass


class SessionClassifier:
    model: torch.nn.Module
    patch_extractor: SessionPatchExtractor
    session: VideoSession
    probability_maps: Dict[int, np.ndarray]
    estimated_locations: Dict[int, np.ndarray]
    result_evaluations: Dict[int, EvaluationResults]

    def __init__(self, video_session, model,
                 patch_size=21,
                 temporal_width=0,

                 mixed_channels=False,
                 drop_confocal=False,

                 standardise=True,
                 to_grayscale=False,

                 n_negatives_per_positive=32,
                 negative_extraction_mode=NegativeExtractionMode.CIRCLE,
                 use_vessel_mask=True,
                 ):
        from copy import deepcopy

        if drop_confocal:
            assert mixed_channels, f'Drop confocal option should only be used with mixed channel option '

        self.model = deepcopy(model)
        self.model = self.model.eval()

        self.session = video_session

        self.standardise = standardise
        self.to_grayscale = to_grayscale

        self._mixed_channels = False
        self._temporal_width = 0

        self.temporal_width = temporal_width

        self.mixed_channels = mixed_channels
        self.drop_confocal = drop_confocal

        self.result_evaluations = {}
        self.probability_maps = {}
        self.estimated_locations = {}

        self.patch_size = patch_size
        self.patch_extractor = SessionPatchExtractor(
            self.session,
            patch_size=patch_size,
            temporal_width=temporal_width,
            extraction_mode=SessionPatchExtractor.ALL_MODE,

            n_negatives_per_positive=n_negatives_per_positive,
            negative_extraction_mode=negative_extraction_mode,
            limit_to_vessel_mask=use_vessel_mask
        )

    def classify_cells(self, frame_idx=None):
        from cnnlearning import LabeledImageDataset

        if frame_idx is None:
            if self.mixed_channels:
                cell_patches = self.patch_extractor.mixed_channel_cell_patches
                non_cell_patches = self.patch_extractor.mixed_channel_non_cell_patches
            elif self.temporal_width > 0:
                cell_patches = self.patch_extractor.temporal_cell_patches_oa790
                non_cell_patches = self.patch_extractor.temporal_non_cell_patches_oa790
            else:
                cell_patches = self.patch_extractor.cell_patches_oa790
                non_cell_patches = self.patch_extractor.non_cell_patches_oa790
        else:
            if self.mixed_channels:
                cell_patches = self.patch_extractor.mixed_channel_cell_patches_at_frame[frame_idx]
                non_cell_patches = self.patch_extractor.mixed_channel_non_cell_patches_at_frame[frame_idx]
            elif self.temporal_width > 0:
                cell_patches = self.patch_extractor.temporal_cell_patches_oa790_at_frame[frame_idx]
                non_cell_patches = self.patch_extractor.temporal_non_cell_patches_oa790_at_frame[frame_idx]
            else:
                cell_patches = self.patch_extractor.cell_patches_oa790_at_frame[frame_idx]
                non_cell_patches = self.patch_extractor.non_cell_patches_oa790_at_frame[frame_idx]

        if self.drop_confocal:
            cell_patches = cell_patches[..., 1:]
            non_cell_patches = non_cell_patches[..., 1:]

        dataset = LabeledImageDataset(
            np.concatenate((cell_patches, non_cell_patches), axis=0),
            np.concatenate(
                (np.ones(len(cell_patches), dtype=np.int32), np.zeros(len(non_cell_patches), dtype=np.int32))),
            standardize=self.standardise, to_grayscale=self.to_grayscale, data_augmentation_transforms=None,
        )
        return classify_labeled_dataset(dataset, model=self.model)

    def estimate_locations(self, frame_idx: int,
                           probability_map: np.ndarray = None,
                           grid_search: bool = False,
                           region_coord_select_mode=RegionCoordSelectMode.GEOMETRIC_CENTROID,
                           extended_maxima_h: float = 0.4,
                           region_max_threshold: float = 0.2,
                           sigma: float = 1.,
                           **patch_extraction_kwargs
                           ) -> np.ndarray:
        """ Estimates the location of the frame

        Args:
            frame_idx (int): The frame index to get the estimated locations
            probability_map (np.ndarray):
             The probability map generated for the frame.
             Must have same shape as frame.
             If not provided then it's calculated using the model.
             If provided then saves time.
            grid_search (bool):
                Whether to perform a grid search on the hyperparameters on the localisation of the estimated cell from
                the probability map.
                This option only works when the frame is marked
            extended_maxima_h:
                Theoextended maxima H for the probability map binarisation.
            region_max_threshold:
                The
            sigma:
            region_coord_select_mode (RegionCoordSelectMode):
                Selection mode from the filtered probability map regions.
            **patch_extraction_kwargs:
                Keyword arguments for patch extraction, check SessionPatchExtractor all_mixed_channel_patches
                and all_patches_oa790

        Returns:

        """
        if grid_search:
            # If the frame is marked then we find the best sigma, H and T that maximise dice's coefficient
            assert self.session.is_marked and frame_idx in self.session.cell_positions, \
                'Grid search option only works when the video has manual markings and the frame specificied: {frame_idx}' \
                ' is marked.'

        if self.mixed_channels:
            patches, mask = self.patch_extractor.all_mixed_channel_patches(frame_idx, ret_mask=True,
                                                                           **patch_extraction_kwargs)
            if self.drop_confocal:
                patches = patches[..., 1:]
        elif self.temporal_width > 0:
            patches, mask = self.patch_extractor.all_temporal_patches(frame_idx, ret_mask=True, **patch_extraction_kwargs)
        else:
            patches, mask = self.patch_extractor.all_patches_oa790(frame_idx, ret_mask=True, **patch_extraction_kwargs)

        if probability_map is None:
            probability_map = create_probability_map(patches, self.model, im_shape=mask.shape, mask=mask,
                                                     standardize=self.standardise, to_grayscale=self.to_grayscale)

        sigmas = [sigma]
        extended_maxima_hs = [extended_maxima_h]
        region_max_thresholds = [region_max_threshold]
        dice_coefficients = []

        # If the frame is marked then we find the best sigma, H and T that maximise dice's coefficient
        if grid_search and self.session.is_marked and frame_idx in self.session.cell_positions:
            sigmas = np.arange(0.2, 2, step=.1)
            extended_maxima_hs = np.arange(0.1, 0.8, step=.1)
            region_max_thresholds = np.arange(0., 0.8, step=.1)

            dice_coefficients = np.zeros((len(sigmas), len(extended_maxima_hs), len(region_max_thresholds)))
            for i, s in enumerate(tqdm(sigmas)):
                for j, h in enumerate(extended_maxima_hs):
                    for k, t in enumerate(region_max_thresholds):
                        estimated_positions = estimate_cell_positions_from_probability_map(
                            probability_map, extended_maxima_h=h,
                            region_coord_select_mode=region_coord_select_mode,
                            region_max_threshold=t,
                            sigma=s)

                        if len(estimated_positions) > 0:
                            evaluation_results = evaluate_results(
                                ground_truth_positions=self.session.cell_positions[frame_idx],
                                estimated_positions=estimated_positions,
                                image=self.session.frames_oa790[frame_idx],
                                mask=mask,
                                patch_size=self.patch_size)
                            dice_coefficients[i, j, k] = evaluation_results.dice

            max_dice_idx = np.argmax(dice_coefficients)
            s_idx, h_idx, t_idx = np.unravel_index(max_dice_idx, dice_coefficients.shape)

            sigma = sigmas[s_idx]
            extended_maxima_h = extended_maxima_hs[h_idx]
            region_max_threshold = region_max_thresholds[t_idx]

        estimated_positions = estimate_cell_positions_from_probability_map(
            probability_map,
            sigma=sigma,
            extended_maxima_h=extended_maxima_h,
            region_max_threshold=region_max_threshold,
            region_coord_select_mode=region_coord_select_mode,
        )

        if frame_idx in self.session.cell_positions:
            result_evaluation = evaluate_results(
                ground_truth_positions=self.session.cell_positions[frame_idx],
                estimated_positions=estimated_positions,
                image=self.session.frames_oa790[self.session.validation_frame_idx],
                mask=mask,
                patch_size=self.patch_size)

            if len(dice_coefficients) == 0:
                dice_coefficients = [result_evaluation.dice]

            result_evaluation.all_sigmas = sigmas
            result_evaluation.all_extended_maxima_hs = extended_maxima_hs
            result_evaluation.all_region_max_thresholds = region_max_thresholds
            result_evaluation.all_dice_coefficients = dice_coefficients

            result_evaluation.probability_map = probability_map
            result_evaluation.sigma = sigma
            result_evaluation.extended_maxima_h = extended_maxima_h
            result_evaluation.region_max_threshold = region_max_threshold
            self.result_evaluations[frame_idx] = result_evaluation

        self.estimated_locations[frame_idx] = estimated_positions
        self.probability_maps[frame_idx] = probability_map

        return estimated_positions

    @property
    def temporal_width(self):
        return self._temporal_width

    @temporal_width.setter
    def temporal_width(self, width):
        if width > 0 and self.mixed_channels:
            raise MutualExclusiveArgumentsException(
                'Temporal width > 0 can not work with mixed channels.'
                'Set mixed channel to False first.')
        self._temporal_width = width

    @property
    def mixed_channels(self):
        return self._mixed_channels

    @mixed_channels.setter
    def mixed_channels(self, mixed_channel_extraction):
        if self.temporal_width > 0 and mixed_channel_extraction:
            raise MutualExclusiveArgumentsException(
                'Mixed channel extraction can not work with temporal width greater than 0.'
                'Set temporal width to 0 first.')
        self._mixed_channels = mixed_channel_extraction

    def save(self, filename, v=False):
        output_file = os.path.join(filename)
        pathlib.Path(pathlib.Path(filename).parent).mkdir(exist_ok=True, parents=True)

        with open(output_file, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            if v:
                print(f'Saved {output_file}')

    @classmethod
    def from_file(cls, file, v=True):
        with open(file, 'rb') as input_file:
            obj = pickle.load(input_file)
            if v and obj:
                print(f'Loaded from', file)
            return obj
