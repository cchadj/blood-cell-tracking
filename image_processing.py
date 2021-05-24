import cv2
import mahotas as mh
import skimage
import torch.utils.data
import torchvision
import tqdm
from skimage.exposure import match_histograms
from skimage.morphology import extrema

import evaluation
from learning_utils import ImageDataset
import numpy as np


def imhmaxima(I, H):
    dtype_orig = I.dtype
    I = np.float32(I)
    return skimage.morphology.reconstruction((I - H), I).astype(dtype_orig)


def equalize_adapt_hist_masked(frame, mask):
    """

    Args:
        frame: Must be np.uint8
        mask:  Must be same shape as frame

    Returns:
        Contrast enhanced image only in the part where it's  not masked
    """
    from patch_extraction import get_mask_bounds
    from skimage.exposure import equalize_adapthist

    processed_frame = np.zeros_like(frame, dtype=np.float32)
    x_min, x_max, y_min, y_max = get_mask_bounds(mask)
    cropped_img = frame[y_min:y_max, x_min:x_max]
    cropped_img = equalize_adapthist(cropped_img)
    processed_frame[y_min:y_max, x_min:x_max] = cropped_img

    return np.uint8(processed_frame * 255)


def equalize_adapt_hist_stack(frames, masks=None):
    from patch_extraction import get_mask_bounds
    from skimage.exposure import equalize_adapthist

    frames = stack_to_masked_array(frames, masks)
    masks = ~frames.mask

    processed_frames = np.ma.empty_like(frames, dtype=np.float64)
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        x_min, x_max, y_min, y_max = get_mask_bounds(mask)
        cropped_img = frame[y_min:y_max, x_min:x_max]
        cropped_img = equalize_adapthist(cropped_img)
        processed_frames[i, y_min:y_max, x_min:x_max] = cropped_img

    return np.uint8(processed_frames * 255)


def imextendedmax(I, H, conn=8):
    if conn == 4:
        structuring_element = np.array([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]],
                                       dtype=np.bool)
    elif conn == 8:
        structuring_element = np.ones([3, 3],
                                      dtype=np.bool)

    h_maxima_result = imhmaxima(I, H)
    extended_maxima_result = mh.regmax(h_maxima_result, Bc=structuring_element)

    return extended_maxima_result


def normalize_data(data, target_range=(0, 1), data_range=None):
    """ Normalizes data to a given target range. (Min Max Normalisation)

    Args:
        data: The data to normalise. Can be of any shape.
        target_range (tuple): (a, b) The target lowest and highest value
        data_range (float): Current data min and max.
                          If None then uses the data minimum and maximum value.
                          Useful when max value may not be observed in the data given.
    Returns:
        Normalised data within target range.
        Same type as data.

    """
    if data_range is None:
        data_range = data.min(), data.max()

    data_min, data_max = data_range
    alpha, beta = target_range

    assert alpha < beta, f'Target range should be from small to big, target range given: {target_range}.'
    assert data_min < data_max, f'Data range should be from small to big, data range given: {data_range}.'

    return ((beta - alpha) * ((data - data_min) / (data_max - data_min)) + alpha).astype(data.dtype)


def hist_equalize(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


def equalize_hist_images(images):
    equalized_images = np.empty_like(images)
    for i, im in enumerate(images):
        equalized_images[i] = hist_equalize(im)[0]

    return equalized_images


def equalize_adapthist_images(images):
    from skimage import exposure
    equalized_images = np.empty_like(images)
    for i, im in enumerate(images):
        equalized_images[i] = np.uint8(exposure.equalize_adapthist(im) * 255)

    return equalized_images


def hist_match_images(images, reference):
    hist_matched_images = np.empty_like(images)

    for i, im in enumerate(tqdm.tqdm(images)):
        hist_matched_images[i] = match_histograms(im, reference)

    return hist_matched_images


def imagestack_to_vstacked_image(framestack):
    return np.hstack(np.split(framestack, len(framestack), axis=0)).squeeze()


def vstacked_image_to_imagestack(stacked_image, n_images):
    return np.array(np.split(stacked_image, n_images, axis=0))


def imagestack_to_hstacked_image(framestack):
    return np.hstack([f.squeeze() for f in np.split(framestack, len(framestack))])


def hstacked_image_to_imagestack(stacked_image, n_images):
    return np.array(np.split(stacked_image, n_images, axis=1))


def hist_match_2(images, reference):
    return vstacked_image_to_imagestack(match_histograms(imagestack_to_vstacked_image(images), reference), len(images))


def frame_differencing(frames, sigma=0):
    import numpy as np
    frames = np.float32(frames.copy())

    background = frames.mean(0)
    if sigma >= 0.125:
        background = mh.gaussian_filter(background, sigma)

    difference_images = np.empty_like(frames)
    for j in range(len(frames) - 1):
        difference_images[j] = mh.gaussian_filter(frames[j], sigma) - background

    return difference_images[:-1]


def crop_mask(mask, left_crop=50):
    """ Crop masks or stack of masks by amount of pixels.

    Can provide a single masks HxW or stack of masks NxHxW
    """

    new_mask = mask.copy()
    if len(new_mask.shape) == 2:
        # If single masks HxW -> 1xHxW
        new_mask = new_mask[np.newaxis, ...]

    for mask_count, m in enumerate(new_mask):
        if np.any(m[:, 0]):
            # edge case where there's a line at the first column in which case we say that the mean edge
            # pixel is the at the first column
            mean_edge_x = 0
        else:
            ys, xs = np.where(np.diff(m))
            edge_ys = np.unique(ys)
            edge_xs = np.empty_like(edge_ys)
            for i, y in enumerate(edge_ys):
                edge_xs[i] = (np.min(xs[np.where(ys == y)[0]]))

            mean_edge_x = np.mean(edge_xs)

        new_mask[mask_count, :, :int(mean_edge_x + left_crop)] = 0

    return new_mask.squeeze()


def stack_to_masked_array(frames, masks=None):
    if not np.ma.is_masked(frames):
        if masks is None:
            masks = np.ones_like(frames, dtype=np.bool8)
        frames = np.ma.masked_array(frames, ~masks)

    return frames


def gaussian_blur_stack(frames, masks=None, sigma=1):
    """

    Args:
        frames (np.ma.masked_array):
            Assumes masked array. Must be N x H x W (x C) of uint8 type
        sigma:

    Returns:

    """
    from skimage import filters
    frames = frames.copy()
    frames = stack_to_masked_array(frames, masks)
    if sigma <= 0.1:
        return frames

    frames = frames / 255
    masks = ~frames.mask

    with np.errstate(divide='ignore', invalid='ignore'):
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            frames[i] = filters.gaussian(frame.filled(0) * mask, sigma=sigma)
            weights = filters.gaussian(mask, sigma=sigma)
            frames[i] /= weights

    return np.uint8(frames * 255)


def enhance_motion_contrast_de_castro(frames, masks=None, sigma=0):
    frames = stack_to_masked_array(frames, masks)

    frames = gaussian_blur_stack(frames, sigma=sigma)
    frames = frames / 255

    for i, frame in enumerate(frames):
        frames[i] /= frame.mean()

    mean_frame = frames.mean(0)
    for i in range(len(frames)):
        frames[i] /= mean_frame
    return frames


def enhance_motion_contrast_mine(frames, masks=None, sigma=1):
    frames = stack_to_masked_array(frames, masks)

    frames = gaussian_blur_stack(frames, sigma=sigma)
    frames /= 255

    for i, frame in enumerate(frames):
        frames[i] /= frame.mean()

    mean_frame = frames.mean(0)
    for i in range(len(frames)):
        frames[i] /= mean_frame

    penultimate_frame = frames[-2].copy()
    for j in range(len(frames) - 1):
        frames[j] /= frames[j + 1]
    frames[-1] /= penultimate_frame

    penultimate_frame = frames[-2].copy()
    frames = frames
    for j in range(len(frames) - 1):
        frames[j] += frames[j + 1]
        frames[j] /= 2

    frames[-1] += penultimate_frame
    frames[-1] /= 2

    return frames


def enhance_motion_contrast_j_tam(frames, masks=None, sigma=0, adapt_hist=False):
    from skimage import exposure

    frames = stack_to_masked_array(frames, masks)

    frames = gaussian_blur_stack(frames, sigma=sigma)

    frames = frames / 255
    print('hey')

    # division frames
    penultimate_frame = frames[-3].copy()
    for j in range(len(frames)):
        if j == len(frames) - 1:
            frame = penultimate_frame
        else:
            frame = frames[j + 1]
        frames[j] /= frame

    # multi-frame division frames
    masks = frames.mask.copy()
    penultimate_frame = frames[-2].copy()
    for j in range(len(frames)):
        if j == len(frames) - 1:
            frame = penultimate_frame
        else:
            frame = frames[j + 1]
        frames[j] += frame
        frames[j] /= 2
        if adapt_hist:
            frame = normalize_data(frames[j].filled(frames[j].mean()))
            frames[j] = exposure.equalize_adapthist(frame)
            frames[j].mask = masks[j]

    return frames


def enhance_motion_contrast(frames, masks=None, method='j_tam', sigma=1, adapt_hist=False):
    if method == 'de_castro':
        return enhance_motion_contrast_de_castro(frames, masks=masks, sigma=sigma)
    elif method == 'j_tam':
        return enhance_motion_contrast_j_tam(frames, masks=masks, sigma=sigma, adapt_hist=adapt_hist)
    else:
        raise NotImplementedError(f'Method {method} not implemented')


def center_crop_images(images, patch_size):
    """ Crops the centre of the stack of images and returns the result

    Args:
        images: NxHxWxC (or NxHxW)
        patch_size (int or tuple): The dimensions of the patch to crop in the middle

    Returns:
      N x patch height x patch width x C ( or Nx patch height x patch width) numpy array
    """
    crop_transform = [torchvision.transforms.CenterCrop(patch_size)]
    dataset = ImageDataset(images, standardize=False, to_grayscale=False, data_augmentation_transforms=crop_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    for batch in loader:
        # NxCxHxW -> NxHxWxC
        center_cropped_images = batch.permute(0, 2, 3, 1).cpu().numpy().squeeze()
        if images.dtype == np.uint8:
            center_cropped_images = np.uint8(center_cropped_images * 255)
        return center_cropped_images


class ImageRegistrator(object):
    def __init__(self, source, target):
        self.source = source
        self.target = target

        # Warp affine doesn't work with boolean
        if source.dtype == np.bool8:
            self.source = np.float32(self.source)
        if target.dtype == np.bool8:
            self.target = np.float32(self.target)

        self.best_dice = evaluation.dice(source, target)
        self.dices = [self.best_dice]
        self.vertical_displacement = 0
        self.horizontal_displacement = 0
        self.registered_source = source

    def register_vertically(self):
        dx = 0
        dys = np.int32(np.arange(1, self.target.shape[0], 1))

        # fig, axes = plt.subplots(len(dys), 1, figsize=(100, 100))

        dices = []
        target_clone = self.target.copy()
        for i, dy in enumerate(dys):
            translation = np.float32([[1, 0, dx],
                                      [0, 1, dy]])

            height, width = self.source.shape[:2]

            translated_source = cv2.warpAffine(self.source, translation, (width, height))

            # Because translating the source introduces black pixels in the first rows (0 to translation)
            # we also zero out the same pixels in the target image to have a more accurate dice coefficient
            target_clone[:dy, :] = 0

            dice_v = evaluation.dice(target_clone, translated_source)
            dices.append(dice_v)

        # Get displacement that gives best dice coefficient.
        dy = dys[np.argmax(dices)]

        translation = np.float32([[1, 0, dx],
                                  [0, 1, dy]])
        height, width = self.source.shape[:2]
        translated_source = cv2.warpAffine(self.source, translation, (width, height))

        self.registered_source = translated_source
        self.vertical_displacement = dy
        self.best_dice = max(dices)
        self.dices = dices

        return self.registered_source

    def apply_registration(self, im):
        dx, dy = self.horizontal_displacement, self.vertical_displacement
        if im.dtype == np.bool:
            im = np.float32(im)

        translation = np.float32([[1, 0, dx],
                                  [0, 1, dy]])
        height, width = im.shape[:2]
        im = cv2.warpAffine(im, translation, (width, height))

        return im

    @staticmethod
    def vertical_image_registration(source, target):
        dx = 0
        dys = np.int32(np.arange(1, 200, 1))

        # fig, axes = plt.subplots(len(dys), 1, figsize=(100, 100))

        dices = []
        for i, dy in enumerate(dys):
            translation = np.float32([[1, 0, dx],
                                      [0, 1, dy]])

            height, width = source.shape[:2]

            translated_source = cv2.warpAffine(source, translation, (width, height))
            dice_v = evaluation.dice(target, translated_source)
            dices.append(dice_v)

        return translated_source, dy


if __name__ == '__main__':
    from shared_variables import get_video_sessions
    import numpy as np

    video_sessions = get_video_sessions(marked=True, registered=True)
    vs = video_sessions[0]
    new_masked_frames = equalize_adapt_hist_stack(vs.masked_frames_oa790)
