import skimage
import skimage.filters
import skimage.exposure
from matplotlib import pylab as plt
import numpy as np
import cv2
from skimage import morphology, measure


def binarize_vessel_image(
        vessel_image,
        normalise_fangi=False,
        equalize_frangi_hist=True,

        threshold_sensitivity=0.5,

        dilation_iterations=0,
        erosion_iterations=0,

        opening_kernel_size=5,
        closing_kernel_size=5,
        padding=cv2.BORDER_REPLICATE,
        padding_size=30,
        padding_value=None,

        visualise_intermediate_steps=False):
    vessel_image = cv2.copyMakeBorder(vessel_image,
                                      padding_size,
                                      padding_size,
                                      padding_size,
                                      padding_size,
                                      padding,
                                      value=padding_value)
    # opening_kernel = morphology.disk(opening_kernel_size)
    # closing_kernel = morphology.disk(closing_kernel_size)
    opening_kernel = morphology.square(opening_kernel_size)
    closing_kernel = morphology.square(closing_kernel_size)

    exposure_iterations = 1
    not_enough_vessel_regions_found = True
    while not_enough_vessel_regions_found:
        vessel_image_contrast_enhanced = vessel_image.copy()

        # exposure iterations increase as long as not enough vessel regions are found, until at least one vessel
        # region is found.
        for _ in range(exposure_iterations):
            sigma = 1
            if sigma > 0:
                vessel_image_contrast_enhanced = skimage.filters.gaussian(vessel_image_contrast_enhanced, sigma=sigma)
            vessel_image_contrast_enhanced = skimage.exposure.equalize_adapthist(vessel_image_contrast_enhanced)

        frangi_image = skimage.filters.frangi(vessel_image_contrast_enhanced, alpha=.5, beta=.5, black_ridges=False)

        if normalise_fangi:
            frangi_image = cv2.normalize(frangi_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        if equalize_frangi_hist:
            frangi_image = skimage.exposure.equalize_adapthist(frangi_image)

        binary_threshold = skimage.filters.threshold_otsu(frangi_image, nbins=256)
        BW = np.zeros_like(frangi_image)
        BW[frangi_image > binary_threshold * threshold_sensitivity] = 1
        BW = np.uint8(BW)

        # Opening get's rid of small speckles or noisy vessel offshoots with small width.
        opening = cv2.morphologyEx(BW, cv2.MORPH_OPEN, opening_kernel)

        opening = skimage.morphology.binary_dilation(opening)
        opening = skimage.morphology.binary_dilation(opening)

        # Closing connects components connects unconnected components and fills small holes.
        closing = cv2.morphologyEx(np.uint8(opening), cv2.MORPH_CLOSE, closing_kernel)

        vessel_mask = closing[padding_size:-padding_size, padding_size:-padding_size]
        # Borders are usually problematic so we remove an amount of pixels
        vessel_mask[:, :20] = 0
        vessel_mask[:, -20:] = 0

        vessel_mask = np.bool8(vessel_mask)
        labeled_image = measure.label(vessel_mask)
        region_props = measure.regionprops(labeled_image)

        region_areas = np.array([prop.area for prop in region_props])
        region_areas = np.delete(region_areas, np.argmax(region_areas))

        not_enough_vessel_regions_found = len(region_areas) <= 1
        exposure_iterations += 1

    final_vessel_mask = morphology.remove_small_objects(vessel_mask, 700)

    # The vessel masks at this stage is not thick enough to account for all blood cells that pass through, so
    # we dilate the final masks.
    for _ in range(dilation_iterations):
        final_vessel_mask = skimage.morphology.binary_dilation(final_vessel_mask)

    for _ in range(erosion_iterations):
        final_vessel_mask = skimage.morphology.binary_erosion(final_vessel_mask)

    if visualise_intermediate_steps:
        fig, axes = plt.subplots(7, 1, figsize=(60, 60))
        no_ticks(axes)
        axes[0].imshow(vessel_image[padding_size:-padding_size, padding_size:-padding_size], cmap='gray')
        axes[0].set_title('Vessel image', fontsize=60)
        axes[1].imshow(vessel_image_contrast_enhanced[padding_size:-padding_size, padding_size:-padding_size],
                       cmap='gray')
        axes[1].set_title('Contrast exposed image', fontsize=60)
        axes[2].imshow(frangi_image[padding_size:-padding_size, padding_size:-padding_size], cmap='hot')
        axes[2].set_title('Frangi image', fontsize=60)
        axes[3].imshow(BW[padding_size:-padding_size, padding_size:-padding_size])
        axes[3].set_title('Binary image', fontsize=60)
        axes[4].imshow(opening[padding_size:-padding_size, padding_size:-padding_size])
        axes[4].set_title('Opening', fontsize=60)
        axes[5].imshow(closing[padding_size:-padding_size, padding_size:-padding_size])
        axes[5].set_title('Closing', fontsize=60)
        axes[6].imshow(final_vessel_mask)
        axes[6].set_title('Removed small components', fontsize=60)
        plt.savefig('intermediate_steps.png')

    return np.bool8(final_vessel_mask)


def create_vessel_image_from_frames(frames, masks=None, de_castro=True, sigma=1, adapt_hist=True):
    from image_processing import enhance_motion_contrast_j_tam, enhance_motion_contrast_de_castro
    from image_processing import stack_to_masked_array, gaussian_blur_stack
    frames = stack_to_masked_array(frames, masks)

    frames = gaussian_blur_stack(frames, sigma=sigma)

    if de_castro:
        frames = enhance_motion_contrast_de_castro(frames, sigma=0)
    frames = enhance_motion_contrast_j_tam(frames, sigma=0, adapt_hist=adapt_hist)

    std_img = frames.std(0)
    std_img = std_img.filled(std_img.mean())

    return std_img


def create_vessel_mask_from_frames(frames, masks=None, vessel_img=None, de_castro=True, sigma=1, adapt_hist=True,
                                   **binarization_kwargs):
    if vessel_img is None:
        vessel_img = create_vessel_image_from_frames(frames, masks, de_castro=de_castro, sigma=sigma, adapt_hist=adapt_hist)

    mask = binarize_vessel_image(vessel_img, **binarization_kwargs)
    return mask


if __name__ == '__main__':
    from plotutils import no_ticks
    from image_processing import ImageRegistrator
    from shared_variables import get_video_sessions
    import matplotlib.pyplot as plt
    import matplotlib.lines

    video_sessions = get_video_sessions(registered=True, marked=True)
    vs = video_sessions[2]

    vessel_mask_oa790 = create_vessel_mask_from_frames(vs.masked_frames_oa790, visualize_intermediate_steps=True)
    vessel_mask_oa850 = create_vessel_mask_from_frames(vs.masked_frames_oa850, visualize_intermediate_steps=True)
    im_registrator = ImageRegistrator(source=vessel_mask_oa850, target=vessel_mask_oa790)

    plt.rcParams['axes.titlesize'] = 15

    fig, axes = plt.subplots(1, 3)
    no_ticks(axes)

    axes[0].imshow(vessel_mask_oa790)
    axes[0].set_title('Vessel masks oa790')

    axes[1].imshow(vessel_mask_oa850)
    axes[1].set_title('Vessel masks oa850')

    axes[2].imshow(im_registrator.register_vertically())
    axes[2].set_title('Registered vessel masks oa850')

    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()

    coord1 = transFigure.transform(axes[0].transData.transform([0, im_registrator.vertical_displacement]))
    coord2 = transFigure.transform(
        axes[2].transData.transform([vessel_mask_oa850.shape[-1], im_registrator.vertical_displacement]))

    line = matplotlib.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                   transform=fig.transFigure)
    fig.lines.append(line)
