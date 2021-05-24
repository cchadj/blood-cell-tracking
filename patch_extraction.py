from typing import Dict

import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import numbers
import enum

from nearest_neighbors import get_nearest_neighbor
from skimage.morphology import binary_erosion
from plotutils import no_ticks

from sklearn.neighbors import KDTree
from video_session import VideoSession


def lerp(point_a, point_b, t):
    direction = point_b - point_a
    return point_a + direction * t


def get_parallel_points(points, distance, npp=3):
    new_points = []

    _, nearest_points_idx = get_nearest_neighbor(points, k=2)

    edges_idx = []
    for idx, nearest_indices in enumerate(nearest_points_idx):
        nbr0_idx = nearest_indices[0]

        if (idx, nbr0_idx) not in edges_idx:
            # edges are always (small_idx, big_idx) to facilitate the detection of duplicate edges
            edges_idx.append((min(idx, nbr0_idx), max(idx, nbr0_idx)))

    # remove edges added twice. For example if edge (0, 1) is added twice keep only one
    edges_idx = list(set(edges_idx))

    for idx0, idx1 in edges_idx:
        point0 = points[idx0]
        point1 = points[idx1]

        edge_vector = point1 - point0
        edge_length = np.linalg.norm(edge_vector)

        normed_edge = edge_vector / edge_length

        perp_edge0 = [edge_vector[1], -edge_vector[0]]
        perp_edge0 = perp_edge0 / np.linalg.norm(perp_edge0)

        perp_edge1 = [-edge_vector[1], edge_vector[0]]
        perp_edge1 = perp_edge1 / np.linalg.norm(perp_edge1)

        parallel_point_0 = np.int32(point0 + perp_edge0 * distance)
        parallel_point_2 = np.int32(point1 + perp_edge0 * distance)

        step_size = 1 / npp
        for i in range(npp):
            t = step_size * i
            new_points.append(lerp(parallel_point_0, parallel_point_2, t))

        parallel_point_0 = np.int32(point0 + perp_edge1 * distance)
        parallel_point_2 = np.int32(point1 + perp_edge1 * distance)

        step_size = 1 / npp
        for i in range(npp):
            t = step_size * i
            new_points.append(lerp(parallel_point_0, parallel_point_2, t))
        new_points.append(lerp(parallel_point_0, parallel_point_2, t))

    new_points = np.array(new_points).squeeze()
    # remove duplicates
    new_points = np.unique(new_points, axis=0)

    kdtree = KDTree(points, metric='euclidean')
    distances, nearest_points = kdtree.query(new_points)
    distances = distances.squeeze()

    # remove new points too close to original points
    distances = distances.squeeze()
    new_points = np.delete(new_points, np.where(distances < distance * .9)[0], axis=0)

    # Remove new points that are too close to each other
    kdtree = KDTree(new_points, metric='euclidean')
    distances, nearest_points = kdtree.query(new_points, k=2)
    distances = distances[:, 1].squeeze()
    new_points = np.delete(new_points, np.where(distances < np.mean(distances) - np.std(distances))[0], axis=0)

    return new_points[:, 0], new_points[:, 1]


def get_perpendicular_points(points, distance, npp=3):
    new_points = []

    _, nearest_points_idx = get_nearest_neighbor(points, k=2)

    edges_idx = []
    for idx, nearest_indices in enumerate(nearest_points_idx):
        nbr0_idx = nearest_indices[0]

        if (idx, nbr0_idx) not in edges_idx:
            edges_idx.append((min(idx, nbr0_idx), max(idx, nbr0_idx)))
            # edges are always (small_idx, big_idx) to facilitate the detection of duplicate edges

    # remove edges added twice
    edges_idx = list(set(edges_idx))
    distances = np.arange(-distance, distance + 1, distance * 2 / npp)

    for idx0, idx1 in edges_idx:
        point0 = points[idx0]
        point1 = points[idx1]

        edge = point1 - point0
        length = np.linalg.norm(edge)

        normed_edge = edge / length

        perp_edge1 = [-edge[1], edge[0]]
        perp_edge1 = perp_edge1 / np.linalg.norm(perp_edge1)

        midpoint = point0 + normed_edge * length * .5

        for d in distances:
            new_points.append(np.int32(midpoint + perp_edge1 * d))

    new_points = np.array(new_points).squeeze()
    # remove duplicates
    new_points = np.unique(new_points, axis=0)

    kdtree = KDTree(points, metric='euclidean')
    distances, nearest_points = kdtree.query(new_points)
    distances = distances.squeeze()
    # remove new points too close to original points
    distances = distances.squeeze()
    new_points = np.delete(new_points, np.where(distances < np.mean(distances) - 4 * np.std(distances))[0], axis=0)

    # Remove new points that are too close to each other
    kdtree = KDTree(new_points, metric='euclidean')
    distances, nearest_points = kdtree.query(new_points, k=2)
    distances = distances[:, 1].squeeze()
    new_points = np.delete(new_points, np.where(distances < np.mean(distances) - 2 * np.std(distances))[0], axis=0)

    return new_points[:, 0], new_points[:, 1]


def get_random_points_on_circles(points, n_points_per_circle=1, max_radius=None, ret_radii=False):
    assert 1 <= n_points_per_circle <= 32, f'Points per circle must be between 1 and 32, not {n_points_per_circle}'

    neighbor_distances, neighbor_idxs = get_nearest_neighbor(points, 2)
    npp = n_points_per_circle - 1

    uniform_angle_displacements = np.array([
        0, np.math.pi,

        1 * .5 * np.math.pi, 3 * np.math.pi * .5,
        1 * np.math.pi * .25, 3 * np.math.pi * .25, 5 * np.math.pi * .25, 7 * np.math.pi * .25,

        1 * .125 * np.math.pi, 3 * .125 * np.math.pi, 5 * .125 * np.math.pi, 7 * .125 * np.math.pi,
        9 * .125 * np.math.pi, 11 * .125 * np.math.pi, 13 * .125 * np.math.pi,

        1 * .0625 * np.math.pi, 3 * .0625 * np.math.pi, 5 * .0625 * np.math.pi, 7 * .0625 * np.math.pi,
        9 * .0625 * np.math.pi, 11 * .0625 * np.math.pi, 13 * .0625 * np.math.pi, 13 * .0625 * np.math.pi,
        15 * .0625 * np.math.pi, 17 * .0625 * np.math.pi, 19 * .0625 * np.math.pi, 21 * .0625 * np.math.pi,
        23 * .0625 * np.math.pi, 25 * .0625 * np.math.pi, 29 * .0625 * np.math.pi, 31 * .0625 * np.math.pi,
    ]).squeeze()
    c = 0
    rxs = np.zeros(len(points) * npp + 2 * len(points), dtype=np.int32)
    rys = np.zeros(len(points) * npp + 2 * len(points), dtype=np.int32)
    radii = np.zeros(len(points))

    for centre_point_idx, (closest_nbr_distances, closest_nbr_indices) in enumerate(
            zip(neighbor_distances, neighbor_idxs)):
        centre_point = points[centre_point_idx]

        prev_vec = None
        # Add the points halfway between the current point and it's closest neighbor points
        for i, (nbr_dist, nbr_idx) in enumerate(zip(closest_nbr_distances, closest_nbr_indices)):
            if nbr_dist / 2 < max_radius:
                nbr_point = points[nbr_idx]
                vec = nbr_point - centre_point
                if i == 1 and prev_vec is not None:
                    if prev_vec @ vec > 0:
                        break
                p = np.int32(centre_point + .5 * vec)
                if p[0] not in rxs or p[1] not in rys:
                    rxs[c], rys[c] = p
                    c += 1

                prev_vec = vec

        # Add random points around a small circle around the point
        r = np.ceil(np.min(closest_nbr_distances) / 2)
        if r > max_radius:
            r = max_radius
        radii[centre_point_idx] = r
        cx, cy = centre_point

        angle = np.random.rand() * np.math.pi * 2
        random_radius_epsilon = 0 * (np.random.rand(len(uniform_angle_displacements)) - 0.5)
        random_angles = np.array(angle + uniform_angle_displacements).squeeze()

        rx = np.array([cx + np.cos(random_angles[:npp]) * (r + random_radius_epsilon[:npp])], dtype=np.int32).squeeze()
        ry = np.array([cy + np.sin(random_angles[:npp]) * (r + random_radius_epsilon[:npp])], dtype=np.int32).squeeze()

        rxs[c:c + npp] = rx
        rys[c:c + npp] = ry

        c += npp

    rxs = rxs[:c]
    rys = rys[:c]

    if ret_radii:
        return rxs, rys, radii
    else:
        return rxs, rys


def get_random_points_on_rectangles(cx, cy, rect_size, n_points_per_rect=1):
    """ Get random points at patch perimeter.

    Args:
        n_points_per_rect: How many points to get on rectangle.
        cx: Rectangle center x component.
        cy: Rectangle center y component.
        rect_size (tuple, int): Rectangle height, width.

    Returns:
        Random points on the rectangles defined by centre cx, cy and height, width

    """
    np.random.seed(0)
    assert type(rect_size) is int or type(rect_size) is tuple
    if type(rect_size) is int:
        rect_size = rect_size, rect_size
    height, width = rect_size
    if type(cx) is int:
        cx, cy = np.array([cx]), np.array([cy])

    assert len(cx) == len(cy)

    rxs = np.zeros(0, dtype=np.int32)
    rys = np.zeros(0, dtype=np.int32)
    for i in range(n_points_per_rect):
        # random number to select edge along which to get the random point (up/down, left/right)
        # (0 or 1 displacement)
        r1 = np.random.rand(len(cx))

        # random  number for swapping displacements
        r2 = np.random.rand(len(cy))

        # Controls 0 or 1 displacement
        t = np.zeros(len(cx))
        t[r1 > 0.5] = 1

        dx = t * width
        dy = np.random.rand(len(cx)) * width

        # print('dx', dx)
        # print('dy', dy)

        dx[r2 > 0.5], dy[r2 > 0.5] = dy[r2 > 0.5], dx[r2 > 0.5]

        # print("r1", r1)
        # print("r2", r2)
        # print("t", t)
        #
        # print('dx', dx)
        # print('dy', dy)

        # if r2 > 0.5:
        #     dy, dx = dx, dy

        rx = (cx - width / 2) + dx
        ry = (cy - height / 2) + dy
        # print(rx.shape)
        # print(ry.shape)

        rxs = np.concatenate((rxs, rx))
        rys = np.concatenate((rys, ry))
        # print(rxs.shape)
        # print(rys.shape)

    return rxs, rys


def get_patch(im, x, y, patch_size):
    """ Get a patch from image

    Args:
        im: The image to get the patch from. HxW or HxWxC
        x (int): The patch center x component (left to right)
        y (int): The patch center y component (top to bot)
        patch_size (tuple): The patch height, width

    Returns:
        height x width x C
    """
    if type(patch_size) == int:
        patch_size = patch_size, patch_size
    height, width = patch_size
    return im[int(y - height / 2):int(y + height / 2),
           int(x - width / 2):int(x + width / 2), ...]


def extract_patches(
        img,
        patch_size=(21, 21),
        padding=cv2.BORDER_REPLICATE,
        padding_value=None,
        mask=None
):
    """
    Extract patches around every pixel of the image.

    To get the row, col coordinates of the ith patch do:

    rows, cols = np.unravel_index(np.arange(patches.shape[0]), frame.shape[0:2])
    # ith patch coordinates
    row, col = rows[i], cols[i]

    Arguments:
        img (np.ndarray): HxWxC or HxW(grayscale) image.
        mask (ndarray):  HxW masks with the valid pixels.
            Patches are extracted only for True masks pixels.
            NO patches are extracted for 0 or False pixels.
            If None then all pixels are considered valid.
        patch_size (tuple): The patch height, width.
        padding:
            'valid' If you want only patches that are entirely inside the image.
                If not valid then one of : [
                    cv2.BORDER_REPLICATE,
                    cv2.BORDER_REFLECT,
                    cv2.BORDER_WRAP,
                    cv2.BORDER_ISOLATED,
                    cv2.BORDER_CONSTANT
                ]
        padding_value:
         The value for the padding in case of cv2.BORDER_CONSTANT.
         If None then uses the mean of the image.

    Returns:
        (np.array):  NxHxWxC
    """
    if isinstance(patch_size, numbers.Number):
        patch_height, patch_width = patch_size, patch_size
    elif isinstance(patch_size, tuple):
        patch_height, patch_width = patch_size

    if mask is None:
        assert mask.shape == img.shape[:2], f'Height and width of masks {mask.shape} must much image {img.shape[:2]}'

    if padding != 'valid':
        if padding_value is None:
            padding_value = img.mean()
        padding_height, padding_width = int((patch_height - 1) / 2), int((patch_width - 1) / 2)
        img = cv2.copyMakeBorder(img,
                                 padding_height,
                                 padding_height,
                                 padding_width,
                                 padding_width,
                                 padding,
                                 padding_value
                                 )
    kernel_height, kernel_width = patch_height, patch_width

    inp = torch.from_numpy(img)

    if len(inp.shape) == 3:
        inp = inp.permute(-1, 0, 1)
    elif len(inp.shape) == 2:
        inp = inp[None, ...]
    inp = inp[None, ...]

    # print("Inp.shape", inp.shape)
    patches = inp.unfold(2, kernel_height, 1).unfold(3, kernel_width, 1)
    # Shape -> 1 x 1 x H x W x Hpatch x Wpatch
    # print("Patches shape 1", patches.shape)

    patches = patches.permute(2, 3, 1, -2, -1, 0)[..., 0]
    # Shape -> H x W x C x Hpatch x Wpatch
    # print("Patches shape 2", patches.shape)
    # patch = patches[80, 28, ...].cpu().numpy()
    # patch = patch.transpose(1, 2, 0)
    # plt.imshow(patch.squeeze())

    patches = patches.contiguous().flatten(0, 1)
    # print("Patches shape 3", patches.shape)
    # Shape -> H*W x C x Hpatch x Wpatch

    patches = patches.permute(0, 2, 3, 1)
    #  Output Shape -> H*W x Hpatch x Wpatch x C
    # print("Patches output shape", patches.shape)

    # ------ To get patch at row=y col=x
    # x, y = 131, 63
    # cone_patch_index = np.ravel_multi_index([[y], [x]], dims=unpadded_image_shape).item()
    # print(cone_patch_index)
    # patch = patches[cone_patch_index, ...].cpu().numpy()
    # plt.imshow(patch.squeeze())

    patches = patches.cpu().numpy()
    if mask is not None:
        mask_flattened = mask.reshape(-1)
        vessel_pixel_indices = np.where(mask_flattened)[0]
        patches = patches[vessel_pixel_indices]

    return patches


def extract_patches_at_positions(
        image,
        positions,
        patch_size=(21, 21),
        padding='valid',
        mask=None,
        visualize_patches=False
):
    """ Extract patches from images at points

    Arguments:
        image: HxW or HxWxC image
        mask: A boolean masks HxW (same height and width as image).
            Only patches inside the masks are extracted.
        positions: shape:(2,) list of (x, y) points. x left to right, y top to bottom
        patch_size (tuple):  Size of each patch.
        padding:
            'valid' If you want only patches that are entirely inside the image.
            If not valid then one of : [
                cv2.BORDER_REPLICATE,
                cv2.BORDER_REFLECT,
                cv2.BORDER_WRAP,
                cv2.BORDER_ISOLATED
            ]

    Returns:
        (np.array): NxHxWxC patches.
    """
    assert 2 <= len(image.shape) <= 3
    if type(patch_size) is tuple:
        patch_height, patch_width = patch_size
    elif type(patch_size) is int:
        patch_height, patch_width = patch_size, patch_size
    else:
        raise TypeError('Patch_size must be int or type. Type given: ', type(patch_size))

    padding_height, padding_width = 0, 0
    n_patches_max = positions.shape[0]

    if padding != 'valid':
        padding_height, padding_width = int((patch_height - 1) / 2), int((patch_width - 1) / 2)
        image = cv2.copyMakeBorder(image,
                                   padding_height,
                                   padding_height,
                                   padding_width,
                                   padding_width,
                                   padding)
        assert positions[:, 1].max() < image.shape[0] and positions[:, 0].max() < image.shape[1], \
            'Position coordinates must not go outside of image boundaries.'
    if len(image.shape) == 2:
        n_channels = 1
        image = image[:, :, np.newaxis]
    elif len(image.shape) == 3:
        n_channels = image.shape[-1]

    patches = np.zeros_like(image, shape=[n_patches_max, patch_height, patch_width, n_channels])

    if visualize_patches:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, figsize=(20, 10))

    if mask is None:
        mask = np.ones_like(image, dtype=np.bool8)

    patch_count = 0
    for x, y in np.int32(positions):
        if not mask[y, x]:
            print('hmm why?')
            continue
        # Offset to adjust for padding
        x, y = x + padding_width, y + padding_height
        patch = get_patch(image, x, y, patch_size)

        #  If patch shape is same as patch size then it means that it's valid
        if patch.shape[:2] == (patch_height, patch_width):
            # print("Heyo ", patches[patch_count].shape)
            patches[patch_count, :, :, :] = patch
            patch_count += 1

            if visualize_patches:
                # Rectangle ( xy -> bottom and left rect coords, )
                # noinspection PyUnresolvedReferences
                rect = matplotlib.patches.Rectangle((x - patch_width / 2,
                                                     y - patch_height / 2),
                                                    patch_width, patch_height, linewidth=1,
                                                    edgecolor='r', facecolor='none')

                ax.imshow(np.squeeze(image), cmap='gray')
                ax.add_patch(rect)
                ax.scatter(x, y)
                ax.annotate(patch_count - 1, (x, y))
        else:
            print(f'Helllooo {patch.shape[:2]}')

    patches = patches[:patch_count, ...]
    return patches.squeeze()


def get_mask_bounds(mask):
    # Add a border in case mask is on border, so np.diff detects when the flip happens
    bordersize = 1
    mask_padded = cv2.copyMakeBorder(
        np.uint8(mask),
        top=bordersize,
        bottom=bordersize,
        left=bordersize + 1,  # The flip is detected one pixel earlier from left to right
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    ).astype(np.bool8)

    ys, xs = np.where(np.diff(mask_padded))
    x_min, x_max = xs.min() - bordersize, xs.max()
    y_min, y_max = ys.min() - bordersize, ys.max()

    return x_min, x_max, y_min, y_max


# Using enum class create enumerations
class NegativeExtractionMode(enum.IntEnum):
    RECTANGLE = 0
    CIRCLE = 1
    PARALLEL = 2
    PERPENDICULAR = 3


class SessionPatchExtractor(object):
    _non_cell_positions: Dict[int, np.ndarray]
    _temporal_width: int

    TRAINING_MODE = 0
    VALIDATION_MODE = 1
    ALL_MODE = 2

    def __init__(
            self,
            session: VideoSession,
            patch_size=21,
            temporal_width=1,

            n_negatives_per_positive=1,
            negative_extraction_mode=NegativeExtractionMode.CIRCLE,
            negative_extraction_radius: int = None,

            limit_to_vessel_mask=False,
            extraction_mode=ALL_MODE,

            v=False,
    ):
        """

        Args:
            session (VideoSession):  The video session to extract patches from
        """
        assert extraction_mode in [SessionPatchExtractor.ALL_MODE, SessionPatchExtractor.VALIDATION_MODE,
                                   SessionPatchExtractor.TRAINING_MODE]
        self._session = session
        self._extraction_mode = extraction_mode
        self._negative_extraction_mode = negative_extraction_mode

        self.v = v
        self.use_vessel_mask = limit_to_vessel_mask
        self._cell_positions = {}
        self._non_cell_positions = {}

        assert type(patch_size) is int or type(patch_size) is tuple

        self.erosion_iterations = 2
        if type(patch_size) is tuple:
            self._patch_size = patch_size
        if type(patch_size) is int:
            self._patch_size = patch_size, patch_size

        self.temporal_width = temporal_width

        self.frame_negative_search_radii = {}

        if negative_extraction_radius is None:
            self._negative_patch_extraction_radius = self._patch_size[0]
        else:
            self._negative_patch_extraction_radius = negative_extraction_radius

        self.n_negatives_per_positive = n_negatives_per_positive

        self._cell_patches_oa790 = None
        self._cell_patches_oa850 = None

        self._marked_cell_patches_oa790 = None
        self._marked_cell_patches_oa850 = None

        self._non_cell_patches_oa790 = None
        self._marked_non_cell_patches_oa790 = None

        self._cell_patches_oa790_at_frame = {}
        self._marked_cell_patches_oa790_at_frame = {}

        self._mixed_channel_cell_patches = None
        self._mixed_channel_non_cell_patches = None

        self._mixed_channel_marked_cell_patches = None
        self._mixed_channel_marked_non_cell_patches = None

        self._mixed_channel_cell_patches_at_frame = {}
        self._mixed_channel_non_cell_patches_at_frame = {}

        self._mixed_channel_marked_cell_patches_at_frame = {}
        self._mixed_channel_marked_non_cell_patches_at_frame = {}

        self._temporal_cell_patches_oa790 = None
        self._temporal_marked_cell_patches_oa790 = None

        self._temporal_non_cell_patches_oa790 = None
        self._temporal_marked_non_cell_patches_oa790 = None

        self._temporal_cell_patches_oa790_at_frame = {}
        self._temporal_marked_cell_patches_oa790_at_frame = {}

        self._temporal_non_cell_patches_oa790_at_frame = {}
        self._temporal_marked_non_cell_patches_oa790_at_frame = {}

        self._non_cell_patches_oa790_at_frame = {}
        self._marked_non_cell_patches_oa790_at_frame = {}

        self._reset_patches()

    def with_session(self, vs: VideoSession):
        self.session = vs

        return self

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, vs: VideoSession):
        if self._session != vs:
            self._reset_patches()
            self._session = vs

    @property
    def extraction_mode(self):
        return self._extraction_mode

    @extraction_mode.setter
    def extraction_mode(self, mode):
        assert self.extraction_mode in [SessionPatchExtractor.VALIDATION_MODE,
                                        SessionPatchExtractor.TRAINING_MODE,
                                        SessionPatchExtractor.ALL_MODE]
        if mode != self.extraction_mode:
            self._reset_patches()
        self._extraction_mode = mode

    @property
    def cell_positions(self, v=False):
        if len(self._cell_positions) == 0:
            for frame_idx, frame_cell_positions in self.session.cell_positions.items():
                if frame_idx >= len(self.session.mask_frames_oa790):
                    break
                mask = self.session.mask_frames_oa790[frame_idx]
                if self.use_vessel_mask:
                    mask = mask & self.session.vessel_mask_oa790

                frame_cell_positions = self._delete_invalid_positions(frame_cell_positions, mask)
                self._cell_positions[frame_idx] = np.int32(frame_cell_positions)

        return self._cell_positions

    @property
    def non_cell_positions(self):
        if len(self._non_cell_positions) == 0:
            if self.v:
                print('Negative extraction mode, ', self._negative_extraction_mode)
            for frame_idx, frame_cell_positions in self.cell_positions.items():
                if frame_idx >= len(self.session.mask_frames_oa790):
                    break

                mask = self.session.mask_frames_oa790[frame_idx]
                if self.use_vessel_mask:
                    # We erode the vessel mask to pick negatives that are mostly parallel to the direction of the flow
                    vessel_mask = self.session.vessel_mask_oa790
                    for _ in range(self.erosion_iterations):
                        vessel_mask = binary_erosion(vessel_mask)
                    mask = mask & vessel_mask

                cx, cy = frame_cell_positions[:, 0], frame_cell_positions[:, 1]

                if self._negative_extraction_mode == NegativeExtractionMode.RECTANGLE:
                    rx, ry = get_random_points_on_rectangles(cx, cy, rect_size=self.negative_patch_extraction_radius,
                                                             n_points_per_rect=self.n_negatives_per_positive)
                elif self._negative_extraction_mode == NegativeExtractionMode.CIRCLE:
                    if len(frame_cell_positions) <= 2:
                        continue

                    rx, ry, radii = get_random_points_on_circles(frame_cell_positions,
                                                                 n_points_per_circle=self.n_negatives_per_positive,
                                                                 max_radius=self.negative_patch_extraction_radius,
                                                                 ret_radii=True,
                                                                 )
                    self.frame_negative_search_radii[frame_idx] = radii
                elif self._negative_extraction_mode == NegativeExtractionMode.PARALLEL:
                    rx, ry = get_parallel_points(frame_cell_positions,
                                                 self.negative_patch_extraction_radius,
                                                 npp=self.n_negatives_per_positive)
                elif self._negative_extraction_mode == NegativeExtractionMode.PERPENDICULAR:
                    rx, ry = get_perpendicular_points(frame_cell_positions,
                                                      self.negative_patch_extraction_radius,
                                                      npp=self.n_negatives_per_positive)

                non_cell_positions = np.array([rx, ry]).T
                non_cell_positions = self._delete_invalid_positions(non_cell_positions, mask)
                non_cell_positions = np.int32(non_cell_positions)
                self._non_cell_positions[frame_idx] = non_cell_positions

        return self._non_cell_positions

    def _visualize_patch_extraction(self,
                                    session_frames,
                                    frame_idx_to_cell_patch_dict,
                                    frame_idx_to_non_cell_patch_dict, masks=None,
                                    frame_idx=None, ax=None, figsize=(50, 40), linewidth=3, s=60, annotate=False):
        """ Shows the patch extraction on the first marked frame that has cell points in it's csv.

        If in Validation mode then shows the validation frame and frame_idx is ignored.
        """
        from plotutils import plot_patch_rois_at_positions, plot_images_as_grid

        if frame_idx is None or frame_idx not in self.cell_positions:
            # Find the first frame index that has marked cell position and patches were extracted
            for frame_idx in list(self.cell_positions.keys()):
                # In case of temporal patches, if frame_idx < temporal width there aren't any patches for that frame
                if frame_idx in frame_idx_to_cell_patch_dict:
                    break

        if self.extraction_mode == SessionPatchExtractor.VALIDATION_MODE:
            frame_idx = self.session.validation_frame_idx

        frame = session_frames[frame_idx]

        mask = None
        if masks is not None:
            mask = masks[frame_idx]

        cell_positions = self.cell_positions[frame_idx]
        cell_positions = self._delete_invalid_positions(cell_positions, mask=mask)
        cell_patches = frame_idx_to_cell_patch_dict[frame_idx]

        non_cell_positions = self.non_cell_positions[frame_idx]
        non_cell_patches = frame_idx_to_non_cell_patch_dict[frame_idx]
        non_cell_positions = self._delete_invalid_positions(non_cell_positions, mask=mask)

        mode_str = ''
        if self.extraction_mode == SessionPatchExtractor.VALIDATION_MODE:
            mode_str = 'Validation'
        if self.extraction_mode == SessionPatchExtractor.TRAINING_MODE:
            mode_str = 'Training'
        if self.extraction_mode == SessionPatchExtractor.ALL_MODE:
            mode_str = 'All'

        negative_extraction_mode_str = ''
        if self._negative_extraction_mode == NegativeExtractionMode.CIRCLE:
            negative_extraction_mode_str = 'Circle negative search'
        elif self._negative_extraction_mode == NegativeExtractionMode.RECTANGLE:
            negative_extraction_mode_str = 'Rectangle negative search'
        elif self._negative_extraction_mode == NegativeExtractionMode.PARALLEL:
            negative_extraction_mode_str = 'Parallel negative search'
        elif self._negative_extraction_mode == NegativeExtractionMode.PERPENDICULAR:
            negative_extraction_mode_str = 'Perpendicular negative search'

        if ax is None:
            self.subplots = plt.subplots(figsize=figsize)
            _, ax = self.subplots

        if mask is None:
            mask = np.ones_like(frame, dtype=np.bool8)

        if self.use_vessel_mask:
            vessel_mask = self.session.vessel_mask_oa790
            for _ in range(self.erosion_iterations):
                vessel_mask = binary_erosion(vessel_mask)
            mask = mask & vessel_mask

        no_ticks(ax)
        ax.imshow(frame * mask, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Frame {frame_idx}. {mode_str} mode. {negative_extraction_mode_str} ', fontsize=20)

        plot_patch_rois_at_positions(cell_positions, self.patch_size, ax=ax, annotate=annotate,
                                     edgecolor='g', pointcolor='g', label='Cell points', linewidth=linewidth)

        plot_patch_rois_at_positions(non_cell_positions, self.patch_size, ax=ax, label='Non cell patches',
                                     annotate=annotate, edgecolor=(1, 0, 0, .35), pointcolor='firebrick',
                                     linewidth=linewidth * 0.75)

        if self._negative_extraction_mode == NegativeExtractionMode.RECTANGLE:
            plot_patch_rois_at_positions(cell_positions, self.negative_patch_extraction_radius, ax=ax,
                                         label='Negative patch extraction radius', linestyle='--',
                                         edgecolor='r', pointcolor='gray', linewidth=linewidth)
        elif self._negative_extraction_mode == NegativeExtractionMode.CIRCLE:
            ax.scatter(non_cell_positions[..., 0], non_cell_positions[..., 1], c='r', s=s)

            for pos, r in zip(cell_positions, self.frame_negative_search_radii[frame_idx]):
                cx, cy = pos
                ax.add_artist(matplotlib.patches.Circle((cx, cy), r, fill=False, edgecolor='r', linewidth=linewidth,
                                                        linestyle='--'))
        else:
            ax.scatter(non_cell_positions[..., 0], non_cell_positions[..., 1], c='r', s=s)

        # plot_images_as_grid(cell_patches, title='Cell patches')
        # plot_images_as_grid(non_cell_patches, title='Non cell patches')
        ax.legend()
        return ax

    def visualize_patch_extraction(self, marked=True, **kwargs):
        if marked:
            frames = self.session.marked_frames_oa790
        else:
            frames = self.session.frames_oa790
        self._visualize_patch_extraction(frames,
                                         self.marked_cell_patches_oa790_at_frame,
                                         self.marked_non_cell_patches_oa790_at_frame,
                                         masks=self.session.mask_frames_oa790,
                                         **kwargs)

    def visualize_temporal_patch_extraction(self, marked=True, **kwargs):
        assert self.temporal_width == 1, 'Visualising temporal patches only works when temporal width is 1'
        if marked:
            frames = self.session.marked_frames_oa790
        else:
            frames = self.session.frames_oa790
        self._visualize_patch_extraction(
            self.session.marked_frames_oa790,
            self.temporal_marked_cell_patches_oa790_at_frame,
            self.temporal_marked_non_cell_patches_oa790_at_frame,
            **kwargs
        )

    def visualize_mixed_channel_patch_extraction(self, channel=1, **kwargs):
        assert channel in [0, 1, 2], f'Channel must be 0 for confocal, 1 for oa850 or 2 for oa790, not{channel}.'
        if channel == 0:
            frames = self.session.frames_confocal
        elif channel == 1:
            frames = self.session.frames_oa790
        elif channel == 2:
            frames = self.session.registered_frames_oa850

        masks = self.session.registered_mask_frames_oa850
        self._visualize_patch_extraction(
            frames,
            self.mixed_channel_marked_cell_patches_at_frame,
            self.mixed_channel_marked_non_cell_patches_at_frame,
            masks=masks,
            **kwargs
        )

    def _reset_positive_patches(self):
        self._cell_patches_oa790 = None
        self._marked_cell_patches_oa790 = None

        self._cell_patches_oa850 = None
        self._marked_cell_patches_oa850 = None

        self._temporal_cell_patches_oa790 = None
        self._temporal_marked_cell_patches_oa790 = None

        self._mixed_channel_cell_patches = None
        self._mixed_channel_marked_cell_patches = None

        self._cell_positions = {}

    def _reset_negative_patches(self):
        self._non_cell_patches_oa790 = None
        self._marked_non_cell_patches_oa790 = None

        self._temporal_non_cell_patches_oa790 = None
        self._temporal_marked_non_cell_patches_oa790 = None

        self._mixed_channel_non_cell_patches = None
        self._mixed_channel_marked_non_cell_patches = None

        self._non_cell_positions = {}

    def _reset_temporal_patches(self):
        self._temporal_cell_patches_oa790 = None
        self._temporal_non_cell_patches_oa790 = None
        self._temporal_marked_cell_patches_oa790 = None
        self._temporal_marked_non_cell_patches_oa790 = None

    def _reset_patches(self):
        """ reset cached patches to create new ones, useful when new patches should be created, i.e when changing video session"""
        self._reset_positive_patches()
        self._reset_negative_patches()
        self._reset_temporal_patches()

    @property
    def patch_size(self):
        return self._patch_size

    @patch_size.setter
    def patch_size(self, patch_size):
        assert type(patch_size) is int or type(patch_size) is tuple

        if type(patch_size) is tuple:
            self._patch_size = patch_size
        if type(patch_size) is int:
            self._patch_size = patch_size, patch_size

        self._reset_positive_patches()

    @property
    def negative_patch_extraction_radius(self):
        return self._negative_patch_extraction_radius

    @negative_patch_extraction_radius.setter
    def negative_patch_extraction_radius(self, radius):
        assert type(radius) == int
        self._negative_patch_extraction_radius = radius
        self._reset_negative_patches()

    @property
    def n_negatives_per_positive(self):
        return self._n_negatives_per_positive

    @n_negatives_per_positive.setter
    def n_negatives_per_positive(self, n):
        self._n_negatives_per_positive = n
        self._reset_negative_patches()

    def all_mixed_channel_patches(self, frame_idx,
                                  use_frame_mask=True,
                                  use_vessel_mask=True,
                                  patch_size=None,
                                  mask=None,
                                  ret_mask=False,
                                  padding=cv2.BORDER_REPLICATE,
                                  padding_value=None):
        """ Extracts a mixed channel patch for every pixel for the frame at frame

        The oa850 video is firstly aligned with the oa790nm video and then the mixed channel patches are extracted.

        A patch of patch_size x patch_size is extracted around every pixel of the image.
        Image is padded so that pixels at the borders also have a patch.

        When use_frame_mask is True then the registered frame mask from the oa850 video is used.
            This is to get only patches that each channel has something and not just zero pixels.

        Args:
            ret_mask (bool):  If True also returns the mask used.
            frame_idx (int): The frame index  of the frame to extract the patches from.
            mask :
             A 2D boolean array with True for the pixels to extract patches from.
             If None then all pixels are valid for patch extraction (unless use_frame_mask or use_vessel mask are True)
            use_frame_mask (bool):
                Set True to use the frame mask provided by the masked video.
                The mask used is the aligned mask from the oa850 mask video.
            use_vessel_mask (bool): Set True to use the vessel mask for the oa790 video session.
            patch_size (int|tuple): The extraction patch size.
            padding: Padding mode for the image. Check opencvs padding modes.
            padding_value:
             The padding value in case cv2.BORDER_CONSTANT is used.
             If None then uses the mean of the frame.

        Returns:
            Nx patch_size x patch_size (x C) Array of the patches extracted.
            If no mask is used  then the number of patches extracted should be N = H * W  where H and W are the
            height and width of the frame.
        """
        if patch_size is None:
            patch_size = self.patch_size

        if mask is None:
            mask = np.ones(self.session.frames_oa790.shape[1:3], dtype=np.bool8)

        if use_vessel_mask:
            mask &= self.session.vessel_mask_oa790

        if use_frame_mask:
            mask &= self.session.registered_mask_frames_oa850[frame_idx]

        patches_confocal = extract_patches(self.session.frames_confocal[frame_idx],
                                           mask=mask,
                                           patch_size=patch_size,
                                           padding=padding,
                                           padding_value=padding_value)

        patches_oa790 = extract_patches(self.session.frames_oa790[frame_idx],
                                        mask=mask,
                                        patch_size=patch_size,
                                        padding=padding,
                                        padding_value=padding_value)

        patches_oa850 = extract_patches(self.session.registered_frames_oa850[frame_idx],
                                        mask=mask,
                                        patch_size=patch_size,
                                        padding=padding,
                                        padding_value=padding_value)

        patches = np.empty((*patches_oa790.shape[:3], 3), dtype=np.uint8)

        patches[..., 0] = patches_confocal.squeeze()
        patches[..., 1] = patches_oa790.squeeze()
        patches[..., 2] = patches_oa850.squeeze()

        if ret_mask:
            return patches, mask
        else:
            return patches

    @property
    def temporal_width(self):
        return self._temporal_width

    @temporal_width.setter
    def temporal_width(self, width):
        assert type(width) is int, f'Temporal width of patch should be an integer not {type(width)}'
        self._temporal_width = width
        self._reset_temporal_patches()

    def _delete_invalid_positions(self,
                                  positions,
                                  mask=None):
        _, frame_height, frame_width = self.session.frames_oa790.shape

        # remove points whose patches get outside the frame
        positions = np.int32(positions)
        positions = np.delete(positions, np.where(positions[:, 0] - np.ceil(self._patch_size[1] / 2) < 0)[0], axis=0)
        positions = np.delete(positions, np.where(positions[:, 1] - np.ceil(self._patch_size[0] / 2) < 0)[0], axis=0)
        positions = np.delete(positions,
                              np.where(positions[:, 0] + np.ceil(self._patch_size[1] / 2) >= frame_width - 1)[0],
                              axis=0)
        positions = np.delete(positions,
                              np.where(positions[:, 1] + np.ceil(self._patch_size[0] / 2) >= frame_height - 1)[0],
                              axis=0)

        if mask is not None and not np.all(mask):
            # remove points whose patches get outside the masks
            x_min, x_max, y_min, y_max = get_mask_bounds(mask)

            # delete points that are outside the mask
            positions = np.delete(positions, np.where(~mask[positions[:, 1], positions[:, 0]])[0], axis=0)

            # delete points too close to the borders of the mask (since the mask is rectified it has borders)
            positions = np.delete(positions, np.where(positions[:, 0] - np.ceil(self._patch_size[1] / 2) < x_min)[0],
                                  axis=0)
            positions = np.delete(positions, np.where(positions[:, 1] - np.ceil(self._patch_size[0] / 2) < y_min)[0],
                                  axis=0)
            positions = np.delete(positions, np.where(positions[:, 0] + np.ceil(self._patch_size[1] / 2) >= x_max)[0],
                                  axis=0)
            positions = np.delete(positions, np.where(positions[:, 1] + np.ceil(self._patch_size[0] / 2) >= y_max)[0],
                                  axis=0)

        return positions

    def _extract_patches_at_positions(self, session_frames, positions, frame_idx_to_patch_dict=None, masks=None):
        """ Extracts patches for each position and frame in points dictionary

        Args:
            session_frames: The frames to extract the patches from
            positions: A DICTIONARY frame index -> N x 2  of points per for frames at each frame index
            frame_idx_to_patch_dict:
                A dictionary to get the patches from that frame.
                Pass an empty dictionary to fill.
            masks:
                The masks of each frame. If position + patch size goes outside the masks the position is discarded.

        Returns:
            Nx patch height x patch width (x C) patches
        """
        patches = np.zeros((0, *self._patch_size), dtype=session_frames.dtype)

        for frame_idx, frame_cell_positions in positions.items():
            if frame_idx >= len(session_frames):
                break

            if self.extraction_mode == SessionPatchExtractor.VALIDATION_MODE and frame_idx != self.session.validation_frame_idx:
                if not self.session.is_validation:
                    continue

            if self.extraction_mode == SessionPatchExtractor.TRAINING_MODE and frame_idx == self.session.validation_frame_idx:
                continue

            if self.extraction_mode == SessionPatchExtractor.TRAINING_MODE and self.session.is_validation:
                continue

            frame = session_frames[frame_idx]
            mask = None
            if masks is not None:
                mask = masks[frame_idx]
                frame_cell_positions = self._delete_invalid_positions(frame_cell_positions, mask)

            cur_frame_cell_patches = extract_patches_at_positions(frame, frame_cell_positions, mask=mask,
                                                                  patch_size=self._patch_size)
            if frame_idx_to_patch_dict is not None:
                frame_idx_to_patch_dict[frame_idx] = cur_frame_cell_patches

            patches = np.concatenate((patches, cur_frame_cell_patches), axis=0)

        return patches

    @property
    def cell_patches_oa790(self):
        if self._cell_patches_oa790 is None:
            self._cell_patches_oa790 = self._extract_patches_at_positions(self.session.frames_oa790,
                                                                          self.cell_positions,
                                                                          self._cell_patches_oa790_at_frame,
                                                                          masks=self.session.mask_frames_oa790)
        return self._cell_patches_oa790

    @property
    def marked_cell_patches_oa790(self):
        if self._marked_cell_patches_oa790 is None:
            self._marked_cell_patches_oa790 = self._extract_patches_at_positions(self.session.marked_frames_oa790,
                                                                                 self.cell_positions,
                                                                                 self._marked_cell_patches_oa790_at_frame,
                                                                                 masks=self.session.mask_frames_oa790)
        return self._marked_cell_patches_oa790

    @property
    def non_cell_patches_oa790(self):
        if self._non_cell_patches_oa790 is None:
            self._non_cell_patches_oa790 = self._extract_patches_at_positions(
                self.session.frames_oa790,
                self.non_cell_positions,
                frame_idx_to_patch_dict=self._non_cell_patches_oa790_at_frame,
                masks=self.session.mask_frames_oa790
            )
        return self._non_cell_patches_oa790

    @property
    def marked_non_cell_patches_oa790(self):
        if self._marked_non_cell_patches_oa790 is None:
            self._marked_non_cell_patches_oa790 = self._extract_patches_at_positions(
                self.session.marked_frames_oa790,
                self.non_cell_positions,
                frame_idx_to_patch_dict=self._marked_non_cell_patches_oa790_at_frame,
                masks=self.session.mask_frames_oa790)

        return self._marked_non_cell_patches_oa790

    @property
    def cell_patches_oa790_at_frame(self):
        # force the cell patches creation ot fill the dict
        tmp = self.cell_patches_oa790

        return self._cell_patches_oa790_at_frame

    @property
    def marked_cell_patches_oa790_at_frame(self):
        # force the cell patches creation to fill the dict
        tmp = self.marked_cell_patches_oa790

        return self._marked_cell_patches_oa790_at_frame

    @property
    def non_cell_patches_oa790_at_frame(self):
        # for the cell patches creation ot fill the dict
        tmp = self.non_cell_patches_oa790
        return self._non_cell_patches_oa790_at_frame

    @property
    def marked_non_cell_patches_oa790_at_frame(self):
        # for the cell patches creation ot fill the dict
        tmp = self.marked_non_cell_patches_oa790
        return self._marked_non_cell_patches_oa790_at_frame

    def all_patches_oa790(self, frame_idx,
                          mask=None,
                          use_frame_mask=True,
                          use_vessel_mask=True,
                          patch_size=None,
                          ret_mask=False,
                          padding=cv2.BORDER_REPLICATE,
                          padding_value=None):
        """ Extracts a patch for every pixel for the frame at frame index from the oa790nm channel frames.

        A patch of patch_size x patch_size is extracted around every pixel of the image.
        Image is padded so that pixels at the borders also have a patch.

        Args:
            frame_idx (int): The frame index  of the frame to extract the patches from.
            mask :
             A 2D boolean array with True for the pixels to extract patches from.
             If None then all pixels are valid for patch extraction (unless use_frame_mask or use_vessel mask are True)
            use_frame_mask (bool): Set True to use the frame mask provided by the masked video.
            use_vessel_mask (bool): Set True to use the vessel mask for the oa790 video session.
            patch_size (int|tuple): The extraction patch size.
            padding: Padding mode for the image. Check opencvs padding modes.
            padding_value:
             The padding value in case cv2.BORDER_CONSTANT is used.
             If None then uses the mean of the frame.
            ret_mask (bool): If True returns the mask used

        Returns:
            Nx patch_size x patch_size (x C) Array of the patches extracted.
            If no mask is used  then the number of patches extracted should be N = H * W  where H and W are the
            height and width of the frame.
        """
        if patch_size is None:
            patch_size = self.patch_size

        if mask is None:
            mask = np.ones(self.session.frames_oa790.shape[1:3], dtype=np.bool8)

        if use_vessel_mask:
            mask &= self.session.vessel_mask_oa790

        if use_frame_mask:
            mask &= self.session.mask_frames_oa790[frame_idx]

        patches = extract_patches(self.session.frames_oa790[frame_idx],
                                  mask=mask,
                                  patch_size=patch_size,
                                  padding=padding,
                                  padding_value=padding_value)
        if ret_mask:
            return patches, mask
        else:
            return patches

    def _extract_temporal_patches(self,
                                  session_frames,
                                  positions,
                                  frame_idx_to_temporal_patch_dict,
                                  masks=None):
        n_channels = 2 * self.temporal_width + 1
        temporal_patches = np.empty((0, *self._patch_size, n_channels), dtype=np.uint8)

        _, frame_height, frame_width = session_frames.shape
        for frame_idx, frame_positions in positions.items():
            if frame_idx >= len(session_frames):
                break

            if frame_idx < self.temporal_width or frame_idx > len(session_frames) - self.temporal_width:
                continue

            if masks is not None:
                mask = masks[frame_idx]

            frame_positions = self._delete_invalid_positions(frame_positions, mask)
            frame_temporal_patches = np.empty((len(frame_positions), *self._patch_size, n_channels), dtype=np.uint8)

            for i, frame in enumerate(
                    session_frames[frame_idx - self.temporal_width:frame_idx + self.temporal_width + 1]):
                extract_patches_at_positions(frame,
                                             frame_positions,
                                             mask=mask,
                                             patch_size=self._patch_size)
                frame_positions = self._delete_invalid_positions(frame_positions, mask)

                frame_temporal_patches[..., i] = extract_patches_at_positions(frame,
                                                                              frame_positions,
                                                                              mask=mask,
                                                                              patch_size=self._patch_size)
            frame_idx_to_temporal_patch_dict[frame_idx] = frame_temporal_patches
            temporal_patches = np.concatenate((temporal_patches, frame_temporal_patches), axis=0)

        return temporal_patches

    @property
    def temporal_cell_patches_oa790(self):
        if self._temporal_cell_patches_oa790 is None:
            self._temporal_cell_patches_oa790_at_frame = {}
            self._temporal_cell_patches_oa790 = self._extract_temporal_patches(
                self.session.frames_oa790,
                self.cell_positions,
                frame_idx_to_temporal_patch_dict=self._temporal_cell_patches_oa790_at_frame,
                masks=self.session.mask_frames_oa790)
        return self._temporal_cell_patches_oa790

    @property
    def temporal_marked_cell_patches_oa790(self):
        if self._temporal_marked_cell_patches_oa790 is None:
            self._temporal_marked_cell_patches_oa790_at_frame = {}
            self._temporal_marked_cell_patches_oa790 = self._extract_temporal_patches(
                self.session.marked_frames_oa790,
                self.cell_positions,
                frame_idx_to_temporal_patch_dict=self._temporal_marked_cell_patches_oa790_at_frame,
                masks=self.session.mask_frames_oa790)
        return self._temporal_marked_cell_patches_oa790

    @property
    def temporal_non_cell_patches_oa790(self):
        if self._temporal_non_cell_patches_oa790 is None:
            self._temporal_non_cell_patches_oa790_at_frame = {}
            self._temporal_non_cell_patches_oa790 = self._extract_temporal_patches(
                self.session.frames_oa790,
                self.non_cell_positions,
                frame_idx_to_temporal_patch_dict=self._temporal_non_cell_patches_oa790_at_frame,
                masks=self.session.mask_frames_oa790)
        return self._temporal_non_cell_patches_oa790

    @property
    def temporal_marked_non_cell_patches_oa790(self):
        if self._temporal_marked_non_cell_patches_oa790 is None:
            self._temporal_marked_non_cell_patches_oa790_at_frame = {}
            self._temporal_marked_non_cell_patches_oa790 = self._extract_temporal_patches(
                self.session.marked_frames_oa790,
                self.non_cell_positions,
                frame_idx_to_temporal_patch_dict=self._temporal_marked_non_cell_patches_oa790_at_frame,
                masks=self.session.mask_frames_oa790,
            )
        return self._temporal_marked_non_cell_patches_oa790

    @property
    def temporal_cell_patches_oa790_at_frame(self):
        tmp = self.temporal_cell_patches_oa790
        return self._temporal_cell_patches_oa790_at_frame

    @property
    def temporal_marked_cell_patches_oa790_at_frame(self):
        tmp = self.temporal_marked_cell_patches_oa790
        return self._temporal_marked_cell_patches_oa790_at_frame

    @property
    def temporal_non_cell_patches_oa790_at_frame(self):
        tmp = self.temporal_non_cell_patches_oa790
        return self._temporal_non_cell_patches_oa790_at_frame

    @property
    def temporal_marked_non_cell_patches_oa790_at_frame(self):
        tmp = self.temporal_marked_non_cell_patches_oa790
        return self._temporal_marked_non_cell_patches_oa790_at_frame

    def all_temporal_patches(self, frame_idx,
                             mask=None,
                             use_frame_mask=True,
                             use_vessel_mask=True,
                             patch_size=None,
                             ret_mask=False,
                             padding=cv2.BORDER_REPLICATE,
                             padding_value=None):
        """ Extracts a patch for every pixel for the frame at frame index from the oa790nm channel frames.

        A patch of patch_size x patch_size is extracted around every pixel of the image.
        Image is padded so that pixels at the borders also have a patch.

        Args:
            frame_idx (int): The frame index  of the frame to extract the patches from.
            mask :
             A 2D boolean array with True for the pixels to extract patches from.
             If None then all pixels are valid for patch extraction (unless use_frame_mask or use_vessel mask are True)
            use_frame_mask (bool): Set True to use the frame mask provided by the masked video.
            use_vessel_mask (bool): Set True to use the vessel mask for the oa790 video session.
            patch_size (int|tuple): The extraction patch size.
            padding: Padding mode for the image. Check opencvs padding modes.
            padding_value:
             The padding value in case cv2.BORDER_CONSTANT is used.
             If None then uses the mean of the frame.
            ret_mask (bool): If True returns the mask used

        Returns:
            Nx patch_size x patch_size (x C) Array of the patches extracted.
            If no mask is used  then the number of patches extracted should be N = H * W  where H and W are the
            height and width of the frame.
        """
        if self.temporal_width > 1:
            raise NotImplementedError(
                f'Currently all temporal patches are extracted for temporal width 1, not {self.temporal_width}')

        if patch_size is None:
            patch_size = self.patch_size

        if mask is None:
            mask = np.ones(self.session.frames_oa790.shape[1:3], dtype=np.bool8)

        if use_vessel_mask:
            mask &= self.session.vessel_mask_oa790

        if use_frame_mask:
            mask &= self.session.mask_frames_oa790[frame_idx]

        patches_frame_0 = extract_patches(self.session.frames_oa790[frame_idx - 1],
                                          mask=mask,
                                          patch_size=patch_size,
                                          padding=padding,
                                          padding_value=padding_value)
        patches_frame_1 = extract_patches(self.session.frames_oa790[frame_idx],
                                          mask=mask,
                                          patch_size=patch_size,
                                          padding=padding,
                                          padding_value=padding_value)
        patches_frame_2 = extract_patches(self.session.frames_oa790[frame_idx + 1],
                                          mask=mask,
                                          patch_size=patch_size,
                                          padding=padding,
                                          padding_value=padding_value)

        patches = np.empty((*patches_frame_0.shape[:3], 3), dtype=np.uint8)

        patches[..., 0] = patches_frame_0.squeeze()
        patches[..., 1] = patches_frame_1.squeeze()
        patches[..., 2] = patches_frame_2.squeeze()

        if ret_mask:
            return patches, mask
        else:
            return patches

    def _extract_mixed_channel_cell_patches(self,
                                            frames_oa790,
                                            positions,
                                            frame_idx_to_patch_dict=None
                                            ):
        frame_idx_to_confocal_patches = {}
        frame_idx_to_oa790_patches = {}
        frame_idx_to_oa850_patches = {}
        cell_patches_confocal = self._extract_patches_at_positions(
            self.session.frames_confocal,
            positions,
            frame_idx_to_patch_dict=frame_idx_to_confocal_patches,
            masks=self.session.registered_mask_frames_oa850
        )
        cell_patches_oa790 = self._extract_patches_at_positions(
            frames_oa790,
            positions,
            frame_idx_to_patch_dict=frame_idx_to_oa790_patches,
            masks=self.session.registered_mask_frames_oa850
        )
        # from skimage.exposure import equalize_adapthist
        # for i, frame in enumerate(self.session.registered_frames_oa850):
        #     self.session.registered_frames_oa850[i] = np.uint8(equalize_adapthist(frame) * 255)

        cell_patches_oa850 = self._extract_patches_at_positions(
            self.session.registered_frames_oa850,
            positions,
            frame_idx_to_patch_dict=frame_idx_to_oa850_patches,
            masks=self.session.registered_mask_frames_oa850
        )
        assert len(cell_patches_oa790) == len(cell_patches_oa850), 'Not the same of patches extracted'

        if frame_idx_to_patch_dict is not None:
            for frame_idx in frame_idx_to_oa850_patches.keys():
                patches_confocal = frame_idx_to_confocal_patches[frame_idx]
                patches_oa790 = frame_idx_to_oa790_patches[frame_idx]
                patches_oa850 = frame_idx_to_oa850_patches[frame_idx]

                mixed_channel_patches = np.empty([*patches_confocal.shape, 3], dtype=frames_oa790.dtype)
                mixed_channel_patches[..., 0] = patches_confocal
                mixed_channel_patches[..., 1] = patches_oa790
                mixed_channel_patches[..., 2] = patches_oa850

                frame_idx_to_patch_dict[frame_idx] = mixed_channel_patches

        mixed_channel_cell_patches = np.empty([*cell_patches_oa790.shape, 3], dtype=frames_oa790.dtype)
        mixed_channel_cell_patches[..., 0] = cell_patches_confocal
        mixed_channel_cell_patches[..., 1] = cell_patches_oa790
        mixed_channel_cell_patches[..., 2] = cell_patches_oa850

        return mixed_channel_cell_patches

    @property
    def all_mixed_channel_cell_patches(self):
        """
        Returns:
            3 channel patches from the confocal video, the oa790 video and the oa850 video.
            The first channel is from the confocal
            second channel is oa790,
            third channel is oa850,
        """
        if self._mixed_channel_cell_patches is None:
            self._mixed_channel_cell_patches_at_frame = {}
            self._mixed_channel_cell_patches = self._extract_mixed_channel_cell_patches(
                self.session.frames_oa790,
                self.cell_positions,
                frame_idx_to_patch_dict=self._mixed_channel_cell_patches_at_frame
            )
        return self._mixed_channel_cell_patches

    @property
    def mixed_channel_cell_patches(self):
        if self._mixed_channel_cell_patches is None:
            self._mixed_channel_cell_patches_at_frame = {}
            self._mixed_channel_cell_patches = self._extract_mixed_channel_cell_patches(
                self.session.frames_oa790,
                self.cell_positions,
                frame_idx_to_patch_dict=self._mixed_channel_cell_patches_at_frame,
            )
        return self._mixed_channel_cell_patches

    @property
    def mixed_channel_marked_cell_patches(self):
        if self._mixed_channel_marked_cell_patches is None:
            self._mixed_channel_marked_cell_patches_at_frame = {}
            self._mixed_channel_marked_cell_patches = self._extract_mixed_channel_cell_patches(
                self.session.marked_frames_oa790,
                self.cell_positions,
                frame_idx_to_patch_dict=self._mixed_channel_marked_cell_patches_at_frame,
            )
        return self._mixed_channel_marked_cell_patches

    @property
    def mixed_channel_non_cell_patches(self):
        if self._mixed_channel_non_cell_patches is None:
            self._mixed_channel_non_cell_patches_at_frame = {}
            self._mixed_channel_non_cell_patches = self._extract_mixed_channel_cell_patches(
                self.session.frames_oa790,
                self.non_cell_positions,
                frame_idx_to_patch_dict=self._mixed_channel_non_cell_patches_at_frame,
            )
        return self._mixed_channel_non_cell_patches

    @property
    def mixed_channel_marked_non_cell_patches(self):
        if self._mixed_channel_marked_non_cell_patches is None:
            self._mixed_channel_marked_non_cell_patches_at_frame = {}
            self._mixed_channel_marked_non_cell_patches = self._extract_mixed_channel_cell_patches(
                self.session.marked_frames_oa790,
                self.non_cell_positions,
                frame_idx_to_patch_dict=self._mixed_channel_marked_non_cell_patches_at_frame,
            )
        return self._mixed_channel_marked_non_cell_patches

    @property
    def mixed_channel_cell_patches_at_frame(self):
        tmp = self.mixed_channel_cell_patches
        return self._mixed_channel_cell_patches_at_frame

    @property
    def mixed_channel_marked_cell_patches_at_frame(self):
        tmp = self.mixed_channel_marked_cell_patches
        return self._mixed_channel_marked_cell_patches_at_frame

    @property
    def mixed_channel_non_cell_patches_at_frame(self):
        tmp = self.mixed_channel_non_cell_patches
        return self._mixed_channel_non_cell_patches_at_frame

    @property
    def mixed_channel_marked_non_cell_patches_at_frame(self):
        tmp = self.mixed_channel_marked_non_cell_patches
        return self._mixed_channel_marked_non_cell_patches_at_frame



if __name__ == '__main__':
    from shared_variables import get_video_sessions
    from matplotlib.colors import NoNorm

    video_sessions = get_video_sessions(marked=True, registered=False)
    video_sessions = [vs for vs in video_sessions if 'shared-videos' not in vs.video_file]
    print([vs.video_file for vs in video_sessions])
    vs = video_sessions[0]

    patch_extractor = SessionPatchExtractor(vs, patch_size=35, temporal_width=1, n_negatives_per_positive=7)

    plt.rcParams['image.cmap'] = 'gray'

    plt.imshow(patch_extractor.temporal_marked_cell_patches_oa790[1, ..., 0], interpolation='none', norm=NoNorm())
    plt.show()

    plt.imshow(patch_extractor.temporal_marked_cell_patches_oa790[1, ..., 1], interpolation='none', norm=NoNorm())
    plt.show()

    plt.imshow(patch_extractor.temporal_marked_cell_patches_oa790[1, ..., 2], interpolation='none', norm=NoNorm())
    plt.show()

    plt.imshow(patch_extractor.temporal_cell_patches_oa790[1, ..., 0], interpolation='none', norm=NoNorm())
    plt.show()

    plt.imshow(patch_extractor.temporal_cell_patches_oa790[1, ..., 1], interpolation='none', norm=NoNorm())
    plt.show()

    plt.imshow(patch_extractor.temporal_cell_patches_oa790[1, ..., 2], interpolation='none', norm=NoNorm())
    plt.show()
    # plot_images_as_grid(patch_extractor.temporal_cell_patches_oa790[:10], title='Temporal cell patches temporal width 1')
    # plot_images_as_grid(patch_extractor.temporal_marked_cell_patches_oa790[:10])
    #
    # plot_images_as_grid(patch_extractor.temporal_non_cell_patches_oa790[:10], title='Temporal non cell patches temporal width 1')

    # plot_images_as_grid(patch_extractor.temporal_marked_non_cell_patches_oa790[:10])
