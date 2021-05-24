import os
import pickle
import copy
import pathlib

import numpy as np
from nearest_neighbors import get_nearest_neighbor_distances
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def get_positions_too_close_to_border(patch_positions, image_shape, patch_size):
    half_patch_height, half_patch_width = np.uint8(np.ceil(patch_size / 2)), np.uint8(np.ceil(patch_size / 2)),

    invalid = np.zeros([len(patch_positions), 1])
    invalid[patch_positions[:, 0] < (1 + half_patch_height)] = 1
    invalid[patch_positions[:, 0] > (image_shape[1] - half_patch_height)] = 1
    invalid[patch_positions[:, 1] < (1 + half_patch_height)] = 1
    invalid[patch_positions[:, 1] > (image_shape[0] - half_patch_height)] = 1

    return np.where(invalid)[0]


def evaluate_results(ground_truth_positions,
                     estimated_positions,
                     image,
                     mask=None,
                     patch_size=(21, 21),
                     visualise_position_comparison=False
                     ):
    """ Evaluate how good are the estimated position in relation to the ground truth.

    Get dice's coefficient, true positive rate and false discovery rate.
    A true positive is when the distance of an estimated position to it's closest ground truth position is less
    than d where d is .75 * median spacing of the ground truth points.

    Args:
        ground_truth_positions (np.array): (Nx2( The ground truth points.
        estimated_positions (np.array): The estimated points.
        image (np.array): The image the cells are in. Used to get the shape for pruning points that are too close to the border.
        patch_size (tuple):  The patch size that the cells are in. Used for pruning points that are too close to the border.
        visualise_position_comparison (bool): If True  plots estimated points superimposed on ground truth points
            over the image.

    Returns:
        (EvaluationResults):
        Dice's coefficient, true positive rate, false discovery rate and other results.
    """
    assert len(ground_truth_positions) > 0
    assert len(estimated_positions) > 0
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=np.bool8)

    ground_truth_positions = ground_truth_positions.astype(np.int32)
    ground_truth_positions_pruned = np.delete(
        ground_truth_positions, np.where(~mask[ground_truth_positions[:, 1], ground_truth_positions[:, 0]])[0], axis=0
    )
    # np.delete(
    #     ground_truth_positions,
    #     get_positions_too_close_to_border(ground_truth_positions, image.shape[:2], patch_size),
    #     axis=0
    # )

    estimated_positions = estimated_positions.astype(np.int32)
    estimated_positions_pruned = np.delete(
        estimated_positions, np.where(~mask[estimated_positions[:, 1], estimated_positions[:, 0]])[0], axis=0
    )
    # np.delete(
    #     estimated_positions,
    #     get_positions_too_close_to_border(estimated_positions, image.shape[:2], patch_size),
    #     axis=0
    # )

    if len(ground_truth_positions_pruned) == 0 or len(estimated_positions_pruned) == 0:
        # return -1, -1, -1 for unexpected output. This happens when all points are near borders.
        return -1, -1, -1

    median_spacing = np.mean(get_nearest_neighbor_distances(ground_truth_positions_pruned))
    distance_for_true_positive = .75 * median_spacing

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(ground_truth_positions_pruned)
    distances_to_closest_ground_truth, indices_of_closest_ground_truth = nbrs.kneighbors(estimated_positions_pruned)
    distances_to_closest_ground_truth, indices_of_closest_ground_truth = distances_to_closest_ground_truth.squeeze(), indices_of_closest_ground_truth.squeeze()
    estimated_to_ground_distances = distances_to_closest_ground_truth.copy()

    # Each prediction is assigned to it's closest manual position.
    # True Positive:
    # If it's distance to the manual position is < distance_for_true_positive then it's considered true positive
    # False Positive:.
    # If it's distance is > distance_for_true_positive then it's considered a  false positive.
    # If multiple cells are assigned to the same manual position, then the closest is selected and the rest are counted
    # as false positive.
    # False Negatives:
    # Manually marked cones that did not have a matching automatically detected cone were considered as false negatives.

    # The indices of the manual points that are assigned to each predicted point
    indices_of_closest_ground_truth = list(indices_of_closest_ground_truth.flatten())

    # The distance of each predicted point to it's closed manual position
    distances_to_closest_ground_truth = list(distances_to_closest_ground_truth.flatten())

    # The estimated point indices
    estimated_indices = list(np.arange(len(indices_of_closest_ground_truth)).flatten())

    # Sorting by indices_of_closest_ground_truth to identify duplicates where for one estimated, 2 ground truth
    # points
    indices_of_closest_ground_truth, distances_to_closest_ground_truth, estimated_indices = zip(*sorted(
        zip(indices_of_closest_ground_truth, distances_to_closest_ground_truth, estimated_indices)))

    # Create dictionary with the manual points and it's matched predicted points.
    # The entries are of type ground_truth_to_estimated_idx_dst[ground_truth_idx] =>
    #                                                         ([dist_to_point_1, dist_to_point2, ...],
    #                                                          [estimated_idx_1, estimated_idx_2, ...])

    estimated_to_ground_truth = {}
    ground_truth_to_estimated = {}

    ground_truth_to_estimated_dst_idx = {}
    for i in range(len(indices_of_closest_ground_truth)):
        closest_ground_truth_point_idx = indices_of_closest_ground_truth[i]

        dist_to_estimated_point = distances_to_closest_ground_truth[i]
        estimated_point_idx = estimated_indices[i]

        estimated_to_ground_truth[estimated_point_idx] = closest_ground_truth_point_idx

        if closest_ground_truth_point_idx in ground_truth_to_estimated_dst_idx.keys():
            ground_truth_to_estimated_dst_idx[closest_ground_truth_point_idx][0].append(dist_to_estimated_point)
            ground_truth_to_estimated_dst_idx[closest_ground_truth_point_idx][1].append(estimated_point_idx)

            ground_truth_to_estimated[closest_ground_truth_point_idx].append(estimated_point_idx)
        else:
            ground_truth_to_estimated_dst_idx[closest_ground_truth_point_idx] = ([dist_to_estimated_point], [estimated_point_idx])

            ground_truth_to_estimated[closest_ground_truth_point_idx] = [estimated_point_idx]

    true_positive_points = np.empty((0, 2), dtype=np.int32)
    true_positive_dists = np.empty(0, dtype=np.float32)

    false_positive_points = np.empty((0, 2), dtype=np.int32)
    false_positive_dists = np.empty(0, dtype=np.float32)

    # By now ground_truth_to_estimated_dst_idx, can have many predicted points for each ground truth position.
    # We keep the estimated position with the smallest distance.
    n_false_positives_duplicates = 0
    for ground_truth_idx in ground_truth_to_estimated_dst_idx.keys():
        dists_to_ground_truth = ground_truth_to_estimated_dst_idx[ground_truth_idx][0]
        matched_estimated_indices = ground_truth_to_estimated_dst_idx[ground_truth_idx][1]

        # find the estimated points that is closest to the ground truth, this index is relative to
        # matched_estimated_indices and dists_to_ground_truth, not all points
        minimum_dist_idx = np.argmin(dists_to_ground_truth)

        # if predicted points that are matched to more than 1 ground truth position,
        # then we increase the number of false positives by the number of extra estimations.
        n_false_positives_duplicates += len(matched_estimated_indices) - 1

        # ground_truth_to_estimated_dst_idx will now contain a single distance and the estimated
        # position index that is assigned to the ground truth position
        ground_truth_to_estimated_dst_idx[ground_truth_idx] = (dists_to_ground_truth[minimum_dist_idx], matched_estimated_indices[minimum_dist_idx])

        ground_truth_point = ground_truth_positions[ground_truth_idx]
        mininum_dist = ground_truth_to_estimated_dst_idx[ground_truth_idx][0]
        minimum_dist_point = estimated_positions_pruned[ground_truth_to_estimated_dst_idx[ground_truth_idx][1]]

        # find the false positive indices
        false_positive_indices = np.delete(matched_estimated_indices, minimum_dist_idx)
        cur_false_positive_points = estimated_positions_pruned[false_positive_indices]
        cur_false_positive_dists = np.delete(dists_to_ground_truth, minimum_dist_idx)

        false_positive_points = np.concatenate((false_positive_points, cur_false_positive_points), axis=0)
        false_positive_dists = np.concatenate((false_positive_dists, cur_false_positive_dists), axis=0)

        if mininum_dist <= distance_for_true_positive:
            true_positive_points = np.concatenate((true_positive_points, minimum_dist_point[None, ...]), axis=0)
            true_positive_dists = np.append(true_positive_dists, mininum_dist)
        else:
            false_positive_points = np.concatenate((false_positive_points, minimum_dist_point[None, ...]), axis=0)
            false_positive_dists = np.append(false_positive_dists, mininum_dist)

    # Final filtering. Every ground truth point has an estimated position.
    # Remove false positives where distance between ground truth and estimated is bigger than .75 * median_spacing
    keys_to_delete = []
    n_false_positives_too_much_distance = 0
    for ground_truth_idx in ground_truth_to_estimated_dst_idx.keys():
        dist_to_estimated_point = ground_truth_to_estimated_dst_idx[ground_truth_idx][0]
        if dist_to_estimated_point >= distance_for_true_positive:
            keys_to_delete.append(ground_truth_idx)
            n_false_positives_too_much_distance += 1

    for ground_truth_idx in keys_to_delete:
        del ground_truth_to_estimated_dst_idx[ground_truth_idx]

    # The remaining entries are true positives.
    # Only the cells with the smallest distance are considered a true positive.
    n_true_positives = len(ground_truth_to_estimated_dst_idx)
    assert n_true_positives == len(true_positive_points)

    # Automatically detected cells that are not matched to a ground truth cell are considered false positives
    # see cunefare cnn paper
    n_false_positives_unmatched_estimations = len(estimated_positions_pruned) - len(ground_truth_to_estimated_dst_idx)
    n_false_positives = n_false_positives_duplicates + n_false_positives_too_much_distance
    assert n_false_positives_unmatched_estimations == n_false_positives,\
        f'The number of unmatched estimated points {n_false_positives_unmatched_estimations} should be the same as the ' \
        f'number of duplicates {n_false_positives_duplicates} and number of points that are too far away from the ground '\
        f'truth point {n_false_positives_too_much_distance}'
    assert n_false_positives == len(false_positive_points)

    # manually marked cells that do not have a matching automatically detected cell are considered as false negatives
    # see cunefare cnn paper
    n_false_negatives = len(ground_truth_positions_pruned) - len(ground_truth_to_estimated_dst_idx)

    n_manual = len(ground_truth_positions_pruned)
    n_automatic = len(estimated_positions_pruned)

    assert n_manual == n_true_positives + n_false_negatives, f'Number of ground truth points {n_manual} should be same' \
                                                             f'as number of true positives {n_true_positives} + ' \
                                                             f'number of false negatives {n_false_negatives} '
    assert n_automatic == n_true_positives + n_false_positives_unmatched_estimations

    true_positive_rate = n_true_positives / n_manual
    false_discovery_rate = n_false_positives / n_automatic
    dices_coefficient = (2 * n_true_positives) / (n_manual + n_automatic)

    results = EvaluationResults(
        dice=dices_coefficient,
        distance_for_true_positive=distance_for_true_positive,

        image=image,
        mask=mask,

        ground_truth_positions=ground_truth_positions_pruned,
        estimated_positions=estimated_positions_pruned,

        estimated_to_ground_truth=estimated_to_ground_truth,
        ground_truth_to_estimated=ground_truth_to_estimated,
        estimated_to_ground_distances=estimated_to_ground_distances,

        true_positive_rate=true_positive_rate,
        false_discovery_rate=false_discovery_rate,

        true_positive_points=true_positive_points,
        false_positive_points=false_positive_points,

        n_true_positives=n_true_positives,
        n_false_positives=n_false_positives,
        n_false_negatives=n_false_negatives,

        true_positive_dists=true_positive_dists,
        false_positive_dists=false_positive_dists,
    )

    if visualise_position_comparison:
        results.visualize()
        plt.show()

    return results


class EvaluationResults:
    true_positive_dists: np.ndarray
    distance_for_true_positive: np.ndarray

    def __init__(self,
                 dice, distance_for_true_positive, ground_truth_positions, true_positive_rate,
                 estimated_to_ground_distances, image, mask,
                 ground_truth_to_estimated, estimated_to_ground_truth,
                 false_discovery_rate, true_positive_dists,
                 false_positive_dists, true_positive_points,
                 false_positive_points, estimated_positions,
                 n_true_positives=None, n_false_positives=None, n_false_negatives=None,
                 ):
        self.extended_maxima_h = None
        self.region_max_threshold = None
        self.sigma = None
        self.probability_map = None

        self.n_true_positives = n_true_positives
        self.n_false_negatives = n_false_negatives
        self.n_false_positives = n_false_positives

        self.image = image
        self.dice = dice
        self.mask = mask
        self.distance_for_true_positive = distance_for_true_positive

        self.ground_truth_to_estimated = ground_truth_to_estimated
        self.estimated_to_ground_truth = estimated_to_ground_truth

        self.ground_truth_positions = ground_truth_positions
        self.estimated_positions = estimated_positions

        self.estimated_to_ground_distances = estimated_to_ground_distances

        self.true_positive_rate = true_positive_rate
        self.false_discovery_rate = false_discovery_rate

        self.true_positive_points = true_positive_points
        self.false_positive_points = false_positive_points

        self.true_positive_dists = true_positive_dists
        self.false_positive_dists = false_positive_dists

        self.all_sigmas = None
        self.all_extended_maxima_hs = None
        self.all_dice_coefficients = None
        self.region_max_thresholds = None

    def visualize(self, show_probability_map=False):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, ConnectionPatch
        _, ax = plt.subplots(figsize=(60, 60))

        ax.imshow(self.image * self.mask, cmap='gray', vmin=0, vmax=255)
        if show_probability_map and self.probability_map is not None:
            ax.imshow(self.probability_map * self.mask, cmap='hot', vmin=0, vmax=1)

        ax.scatter(self.ground_truth_positions[:, 0], self.ground_truth_positions[:, 1],
                   c='blue', s=230, label='Ground truth points')

        for point in self.ground_truth_positions:
            circ = Circle(point, self.distance_for_true_positive, fill=False, linestyle='--', color='blue')
            ax.add_artist(circ)

        ax.scatter(self.estimated_positions[:, 0], self.estimated_positions[:, 1],
                   c='yellow', label='Predicted Positions')

        for estimated_idx, ground_truth_idx in self.estimated_to_ground_truth.items():
            estimated_point = self.estimated_positions[estimated_idx]
            ground_truth_point = self.ground_truth_positions[ground_truth_idx]
            dist_to_ground_truth = self.estimated_to_ground_distances[estimated_idx]

            if dist_to_ground_truth <= self.distance_for_true_positive:
                con = ConnectionPatch(estimated_point, ground_truth_point, 'data', 'data',
                                      arrowstyle='-', shrinkA=5, shrinkB=5, mutation_scale=20, fc="w")
                ax.add_artist(con)

        ax.scatter(self.false_positive_points[:, 0], self.false_positive_points[:, 1],
                   c='red', s=150, label='False positive points')

        ax.scatter(self.true_positive_points[:, 0], self.true_positive_points[:, 1],
                   c='green', s=200, label='True positive points')

        ax.set_title(f"Dice Coefficient {self.dice:.3f}.\n"
                     f'Distance between ground truth point and estimated point must be less than {self.distance_for_true_positive:.3f} to be TP.\n'
                     f'Mean true positive distance {self.true_positive_dists.mean():3f}\n'
                     f'True positive rate {self.true_positive_rate:3f}.\n'
                     f'False discovery rate {self.false_discovery_rate:3f}.\n')
        ax.legend()

    def save(self, filename, v=False):
        output_file = os.path.join(filename)
        pathlib.Path(pathlib.Path(filename).parent).mkdir(exist_ok=True, parents=True)

        with open(output_file, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            if v:
                print(f'Saved {output_file}')

    @classmethod
    def from_file(cls, file):
        with open(file, 'rb') as input_file:
            return pickle.load(input_file)


def dice(mask1, mask2):
    """ Computes the Dice coefficient, a measure of set similarity.

    Implementation thanks to: https://gist.github.com/JDWarner/6730747

    Parameters
    ----------
    mask1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    mask2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    mask1 = np.asarray(mask1).astype(np.bool)
    mask2 = np.asarray(mask2).astype(np.bool)

    if mask1.shape != mask2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(mask1, mask2)

    return 2 * intersection.sum() / (mask1.sum() + mask2.sum())
