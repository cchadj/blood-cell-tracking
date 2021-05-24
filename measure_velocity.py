from cnnlearning import *
from learning_utils import *
from patch_extraction import *
from image_processing import *
from nearest_neighbors import *
from evaluation import *
from classificationutils import *
from shared_variables import *
from vessel_detection import *
from generate_datasets import *
from plotutils import no_ticks, plot_images_as_grid
from guitools import CvRoipolySelector, CvPointSelector

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import os

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 2000)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# size=25
size = 10
params = {'legend.fontsize': 'large',
          'figure.figsize': (20, 8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size * 0.75,
          'ytick.labelsize': size * 0.75,
          'axes.titlepad': 25}
plt.rcParams.update(params)


def get_positions_from_csv(csv_file, frame_idx):
    df = pd.read_csv(csv_file)
    all_cell_positions = df[['X', 'Y']].to_numpy().astype(np.int32)
    all_cell_frame_indices = df[['Slice']].to_numpy().astype(np.int32)

    return all_cell_positions[np.where(all_cell_frame_indices == frame_idx)[0]]


video_sessions = get_video_sessions(marked=True)
# keep only the video sessions that have vessel masks
video_sessions = [vs for vs in video_sessions if vs.vessel_mask_confocal_file and vs.vessel_mask_oa850_file]

assert len(video_sessions) == 8
assert all([vs.vessel_mask_confocal_file for vs in video_sessions])
assert all([vs.vessel_mask_oa850_file for vs in video_sessions])
assert all([vs.video_oa790_file for vs in video_sessions])
assert all([vs.marked_video_oa790_file for vs in video_sessions])
assert all([len(vs.cell_position_csv_files) > 0 for vs in video_sessions])

vs = video_sessions[0]
# To see all the attributes of the VideoSession object do:
# vs.__dict__


def match_template(image, template,
                   method=cv2.TM_CCOEFF_NORMED):
    template_matching_methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                                 cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, 'template_match']
    assert method in template_matching_methods

    template = np.float32(template)
    image = np.float32(image)

    template_w, template_h = template.shape
    res = cv2.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    cx, cy = (np.floor(top_left[0] + template_w / 2), np.floor(top_left[1] + template_h / 2))
    plt.imshow(res)
    return (cx, cy), res


def find_bloodcell_correspondance(im_790, im_850, method='template_match', ret_correlation_im=False):
    im_790 = np.float32(im_790)
    im_850 = np.float32(im_850)

    template_matching_methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                                 cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED,
                                 'template_match']
    feature_matching_methods = ['surf']

    assert method in template_matching_methods or method in feature_matching_methods

    correlation_im = None
    if method in template_matching_methods:
        if method == 'template_match':
            # default template matching method is cross corellation
            method = cv2.TM_CCORR_NORMED

        h, w = im_790.shape
        centre_row, centre_col = int(h / 2), int(w / 2)
        template_cell = im_790[centre_row - 11:centre_row + 11,
                        centre_col - 11:centre_col + 11]
        (matched_x, matched_y), correlation_im = match_template(im_850, template_cell, method)
    elif method in feature_matching_methods:
        if method == 'surf':
            im_790 = np.uint8(im_790)
            im_890 = np.uint8(im_850)
            for i in range(400, 0, -10):
                # Initiate ORB detector
                surf = cv2.xfeatures2d.SURF_create(i)

                # find the keypoints and descriptors with ORB
                kp1, des1 = surf.detectAndCompute(im_790, None)
                kp2, des2 = surf.detectAndCompute(im_890, None)

                if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
                    continue

                # create BFMatcher object
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                # Match descriptors.
                matches = bf.match(des1, des2)

                if len(matches) == 0:
                    continue
                # Sort them in the order of their distance.
                matches = sorted(matches, key=lambda x: x.distance)
                # Draw first match
                match = matches[0]
                img1_idx = match.queryIdx
                img2_idx = match.trainIdx

                # x - columns
                # y - rows
                # Get the coordinates
                x1, y1 = kp1[img1_idx].pt
                matched_x, matched_y = kp2[img2_idx].pt
                break
    else:
        raise Exception(
            f'No such method {method} for bloodcell correspondance.\n Use one of {template_matching_methods.extend(feature_matching_methods)}')

    if ret_correlation_im:
        return matched_x, matched_y, correlation_im
    else:
        return matched_x, matched_y


def create_average_images(video_session, patch_size=51, sigma=1, average_all_frames=True):
    # Input
    frame_idx = 0
    vs = video_session


    vessel_mask_confocal = vs.vessel_mask_confocal
    vessel_mask_oa850 = vs.vessel_mask_oa850


    image_registrator = ImageRegistrator(source=vessel_mask_oa850, target=vessel_mask_confocal)
    image_registrator.register_vertically()

    registered_frames_oa850 = vs.frames_oa850.copy()

    for i, frame in enumerate(registered_frames_oa850):
        registered_frames_oa850[i] = image_registrator.apply_registration(frame)

    frame_oa790 = vs.frames_oa790[frame_idx]
    frame_oa850 = vs.frames_oa850[frame_idx]
    marked_frame_oa790 = vs.marked_frames_oa790[frame_idx]
    # plot
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['figure.titlesize'] = 30

    #### #### #### #### #### #### #### #### #### ####

    # cv2.imshow(window, frame_OA790)
    registered_frame_oa850 = image_registrator.apply_registration(frame_oa850)
    registered_frame_oa850 = np.concatenate((registered_frame_oa850[..., None],
                                             registered_frame_oa850[..., None],
                                             registered_frame_oa850[..., None]), axis=-1)
    marked_frame_oa790 = cv2.cvtColor(marked_frame_oa790, cv2.COLOR_GRAY2RGB)
    # all_cell_positions = np.empty((0, 2), dtype=np.int32)
    # for frame_idx, points in vs.points.items():
    #     all_cell_positions = np.concatenate((all_cell_positions, points), axis=0)

    for x, y in vs.cell_positions[frame_idx]:
        marked_frame_oa790 = cv2.circle(marked_frame_oa790, (x, y), 1, (255, 0, 0))
    marked_frame_oa790[:image_registrator.vertical_displacement, :] = 0
    window = 'select segment'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    roipoly_selector = CvRoipolySelector(window, marked_frame_oa790)
    roipoly_selector.activate()

    selected_vessel_mask = roipoly_selector.mask.copy()
    plt.imshow(frame_oa790 * selected_vessel_mask)

    patch_size_oa850 = patch_size + 20
    all_cell_patches_oa790 = np.empty((0, patch_size, patch_size))
    all_cell_patches_oa850 = np.empty((0, patch_size_oa850, patch_size_oa850))
    for frame_idx, frame_cell_positions in vs.cell_positions.items():
        frame_oa790 = vs.frames_oa790[frame_idx]
        frame_oa850 = vs.frames_oa850[frame_idx]

        cell_patches_oa790 = extract_patches_at_positions(frame_oa790, frame_cell_positions,
                                                          mask=selected_vessel_mask,
                                                          patch_size=patch_size)
        cell_patches_oa850 = extract_patches_at_positions(image_registrator.apply_registration(frame_oa850),
                                                          frame_cell_positions,
                                                          mask=selected_vessel_mask,
                                                          patch_size=patch_size_oa850)

        all_cell_patches_oa790 = np.concatenate((all_cell_patches_oa790, cell_patches_oa790))
        all_cell_patches_oa850 = np.concatenate((all_cell_patches_oa850, cell_patches_oa850))
        if not average_all_frames:
            break

    for i, (cell_patch_oa790, cell_patch_oa850) in enumerate(zip(all_cell_patches_oa790, all_cell_patches_oa850)):
        all_cell_patches_oa790[i] = mh.gaussian_filter(cell_patch_oa790, sigma)
        all_cell_patches_oa850[i] = mh.gaussian_filter(cell_patch_oa850, sigma)

    avg_cell_oa790 = np.average(all_cell_patches_oa790, axis=0)
    avg_cell_oa850 = np.average(all_cell_patches_oa850, axis=0)

    # plot
    plot_images_as_grid(cell_patches_oa790)
    plot_images_as_grid(cell_patches_oa850)

    avg_cell_oa790_clone = avg_cell_oa790.copy()
    avg_cell_oa850_clone = avg_cell_oa850.copy()

    # The template is in the middle of the average cell oa790
    h, w = avg_cell_oa790.shape
    centre_row, centre_col = int(h / 2), int(w / 2)
    template_cell = avg_cell_oa790[centre_row - 11:centre_row + 11,
                    centre_col - 11:centre_col + 11]

    cv2.rectangle(avg_cell_oa790_clone, (centre_row - 11, centre_col - 11), (centre_row + 11, centre_col + 11),
                  color=(255, 0, 0))
    (cx, cy), _ = match_template(avg_cell_oa790, template_cell)
    (cx_850, cy_850), corellation_image = match_template(avg_cell_oa850, template_cell)

    template = template_cell.astype(np.float32)
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED',
               'surf', 'per-patch-template-matching']
    axes: np.ndarray
    fig, axes = plt.subplots(len(methods), 3)
    for i, meth in enumerate(methods):
        method = meth
        if meth != 'surf' and method != 'per-patch-template-matching':
            method = eval(meth)

        try:
            if method == 'per-patch-template-matching':
                xs_ys = np.zeros((len(cell_patches_oa790), 2), dtype=np.int32)

                fig2, per_template_axes = plt.subplots(len(cell_patches_oa790), 2, figsize=(50, 50))
                fig2.suptitle('Per patch template matching')
                for j, (patch_oa790, patch_oa850) in enumerate(zip(cell_patches_oa790, cell_patches_oa850)):
                    cx, cy = find_bloodcell_correspondance(patch_oa790, patch_oa790, method=cv.TM_CCORR_NORMED)
                    x, y, correlation_im = find_bloodcell_correspondance(patch_oa790, patch_oa850,
                                                                         method=cv.TM_CCORR_NORMED, ret_correlation_im=True)
                    xs_ys[j] = np.array((x, y))
                    per_template_axes[j, 0].imshow(patch_oa790)
                    per_template_axes[j, 0].scatter(cx, cy)
                    per_template_axes[j, 1].imshow(patch_oa850)
                    per_template_axes[j, 1].scatter(x, y)

                mean_x_y = np.mean(xs_ys, axis=1)
                x, y = mean_x_y[0], mean_x_y[1]
            else:
                x, y, correlation_im = find_bloodcell_correspondance(avg_cell_oa790, avg_cell_oa850, method,
                                                                     ret_correlation_im=True)
            print(f'Method {meth} ({x, y})')
        except Exception as e:
            print(f'Method {meth} failed {e}')

        try:
            cx, cy = find_bloodcell_correspondance(avg_cell_oa790, avg_cell_oa790, method)
        except:
            print(f'Method {meth} failed')

        no_ticks(axes)
        axes[i, 0].imshow(avg_cell_oa790, cmap='gray')
        axes[i, 0].set_title(f'Detected Point oa790 {avg_cell_oa790.shape}')
        axes[i, 0].scatter(cx, cy)

        axes[i, 1].scatter(cx, cy)
        axes[i, 1].imshow(avg_cell_oa850, cmap='gray')
        axes[i, 1].set_title(f'Detected Point oa850 {avg_cell_oa850.shape}')
        axes[i, 1].scatter(x, y)

        if correlation_im is not None:
            print(avg_cell_oa790.shape)
            print(correlation_im.shape)

            axes[i, 2].imshow(correlation_im, cmap='gray')
            axes[i, 2].scatter(x, y)
            axes[i, 2].set_title(f'correlation image oa850 {correlation_im.shape}')
        fig.suptitle(meth)
    plt.show()
    for i, patch in enumerate(all_cell_patches_oa790):
        Image.fromarray(patch.astype(np.uint8)).save(f'cell-patch-{i}.png')


if __name__ == '__main__':
    create_average_images(get_video_sessions(marked=True)[2], patch_size=51, sigma=0.2, average_all_frames=False)