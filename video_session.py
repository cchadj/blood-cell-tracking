import warnings
import os
import re
from typing import List, Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from os.path import basename

from videoutils import get_frames_from_video
from collections import OrderedDict


class VideoSession(object):
    """ A VideoSession represents the Confocal, oa790, oa850 video files.

    Use to retrieve:
        * the video file names
        * the frames for each video of the same session as a numpy array.
        * the oa850 channel frames registered to the oa790 channel (using the corresponding vessel masks)
        * the vessel masks of each video
        * the std deviation image of each video
        * the csv(s) with the cell locations

    """
    import image_processing
    _image_registrator: image_processing.ImageRegistrator

    def __init__(self, video_filename, load_vessel_mask_from_file=True):
        from shared_variables import find_filename_of_same_source, files_of_same_source
        from shared_variables import marked_video_oa790_files, marked_video_oa850_files
        from shared_variables import csv_cell_cords_oa790_filenames
        from shared_variables import unmarked_video_oa790_filenames, unmarked_video_oa850_filenames
        from shared_variables import unmarked_video_confocal_filenames, mask_video_oa790_files
        from shared_variables import mask_video_oa850_files, mask_video_confocal_files
        from shared_variables import vessel_mask_confocal_files, vessel_mask_oa790_files, vessel_mask_oa850_files
        from shared_variables import std_image_confocal_files, std_image_oa790_files, std_image_oa850_files
        import pathlib

        self._image_registrator = None

        self.is_registered = '_reg_' in video_filename
        self.is_validation = 'validation' in pathlib.PurePath(video_filename).parts

        self.load_vessel_mask_from_file = load_vessel_mask_from_file
        vid_marked_790_filename = find_filename_of_same_source(video_filename, marked_video_oa790_files)
        vid_marked_850_filename = find_filename_of_same_source(video_filename, marked_video_oa850_files)

        self._cell_position_csv_files = [csv_file for csv_file
                                         in csv_cell_cords_oa790_filenames
                                         if files_of_same_source(csv_file, video_filename)]

        self.has_marked_video = vid_marked_790_filename != '' or vid_marked_850_filename != ''
        self.has_marked_cells = len(self.cell_position_csv_files) > 0
        self.is_marked = len(self.cell_position_csv_files) > 0

        self.subject_number = -1
        self.session_number = -1
        for string in video_filename.split('_'):
            if 'Subject' in string:
                self.subject_number = int(re.search(r'\d+', string).group())
            if 'Session' in string:
                self.session_number = int(re.search(r'\d+', string).group())

        self._validation_frame_idx = None

        self.video_file = video_filename

        self.filename = self.video_file
        self.basename = os.path.join(f'{os.path.splitext(os.path.basename(self.video_file))[0]}')
        self.uid = int(self.basename.split('_')[5])

        self.video_oa790_file = find_filename_of_same_source(video_filename, unmarked_video_oa790_filenames)
        self.video_oa850_file = find_filename_of_same_source(video_filename, unmarked_video_oa850_filenames)
        self.video_confocal_file = find_filename_of_same_source(video_filename, unmarked_video_confocal_filenames)

        self.marked_video_oa790_file = vid_marked_790_filename
        self.marked_video_oa850_file = vid_marked_850_filename

        self.mask_video_oa790_file = find_filename_of_same_source(video_filename, mask_video_oa790_files)
        self.mask_video_oa850_file = find_filename_of_same_source(video_filename, mask_video_oa850_files)
        self.mask_video_confocal_file = find_filename_of_same_source(video_filename, mask_video_confocal_files)

        self.vessel_mask_oa790_file = find_filename_of_same_source(video_filename, vessel_mask_oa790_files)
        self.vessel_mask_oa850_file = find_filename_of_same_source(video_filename, vessel_mask_oa850_files)
        self.vessel_mask_confocal_file = find_filename_of_same_source(video_filename, vessel_mask_confocal_files)

        self.std_image_oa790_file = find_filename_of_same_source(video_filename, std_image_oa790_files)
        self.std_image_oa850_file = find_filename_of_same_source(video_filename, std_image_oa850_files)
        self.std_image_confocal_file = find_filename_of_same_source(video_filename, std_image_confocal_files)

        self._frames_oa790 = None
        self._frames_oa850 = None
        self._frames_confocal = None

        self._registered_frames_oa850 = None
        self._registered_mask_frames_oa850 = None
        self._registered_vessel_mask_oa850 = None

        self._mask_frames_oa790 = None
        self._mask_frames_oa850 = None
        self._mask_frames_confocal = None

        self._masked_frames_oa790 = None
        self._masked_frames_oa850 = None
        self._masked_frames_confocal = None

        self._vessel_masked_frames_oa790 = None
        self._vessel_masked_frames_oa850 = None
        self._vessel_masked_frames_confocal = None

        self._fully_masked_frames_oa790 = None
        self._fully_masked_frames_oa850 = None
        self._fully_masked_frames_confocal = None

        self._marked_frames_oa790 = None
        self._marked_frames_oa850 = None

        self._cell_positions = {}

        self._std_image_oa790 = None
        self._std_image_oa850 = None
        self._std_image_confocal = None

        self._vessel_mask_oa790 = None
        self._vessel_mask_oa850 = None
        self._vessel_mask_confocal = None

    def save_vessel_masks(self, v=False):
        import PIL.Image

        output_file_name = f'{os.path.splitext(self.video_oa790_file)[0]}_vessel_mask.png'
        PIL.Image.fromarray(np.uint8(self.vessel_mask_oa790 * 255)).save(output_file_name)
        if v:
            print('saved: ', output_file_name)

        output_file_name = f'{os.path.splitext(self.video_oa850_file)[0]}_vessel_mask.png'
        PIL.Image.fromarray(np.uint8(self.vessel_mask_oa850 * 255)).save(output_file_name)
        if v:
            print('saved: ', output_file_name)

        output_file_name = f'{os.path.splitext(self.video_confocal_file)[0]}_vessel_mask.png'
        PIL.Image.fromarray(np.uint8(self.vessel_mask_confocal * 255)).save(output_file_name)
        if v:
            print('saved: ', output_file_name)

    @staticmethod
    def _write_video(frames, filename, fps=24):
        import os
        import pathlib
        import cv2
        directory, _ = os.path.split(filename)
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        height, width = frames.shape[1:3]
        vid_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

        for frame in frames:
            frame = frame[..., np.newaxis]
            frame = np.concatenate((frame, frame, frame), axis=-1)

            vid_writer.write(frame)
        vid_writer.release()

    def write_video_oa790(self, filename, fps=24, masked=False):
        if masked:
            VideoSession._write_video(self.frames_oa790 * self.vessel_mask_oa790, filename, fps)
        else:
            VideoSession._write_video(self.frames_oa790, filename, fps)

    def write_video_oa850(self, filename, fps=24):
        VideoSession._write_video(self.frames_oa850, filename, fps)

    def write_video_confocal(self, filename, fps=24):
        VideoSession._write_video(self.frames_confocal, filename, fps)

    @staticmethod
    def _assert_frame_assignment(old_frames, new_frames):
        assert new_frames.shape[1:3] == old_frames.shape[1:3], \
            f'Assigned frames should have the same height and width. Old dims {old_frames.shape[1:3]} new dims {new_frames.shape[1:3]}'
        assert new_frames.dtype == old_frames.dtype, \
            f'Assigned frames should have the same type. Old type new {old_frames.dtype} type {new_frames.dtype}'
        assert len(new_frames.shape) == 3, f'The assigned frames should be grayscale (shape given {new_frames.shape})'

    @property
    def frames_oa790(self):
        if self._frames_oa790 is None:
            self._frames_oa790 = get_frames_from_video(self.video_oa790_file)[..., 0]
        return self._frames_oa790

    @frames_oa790.setter
    def frames_oa790(self, new_frames):
        # VideoSession._assert_frame_assignment(self.frames_oa790, new_frames)

        self._masked_frames_oa790 = None
        self._vessel_masked_frames_oa790 = None
        self._fully_masked_frames_oa790 = None

        self._frames_oa790 = new_frames

    @property
    def frames_oa850(self):
        if self._frames_oa850 is None:
            self._frames_oa850 = get_frames_from_video(self.video_oa850_file)[..., 0]
        return self._frames_oa850

    @frames_oa850.setter
    def frames_oa850(self, new_frames):
        VideoSession._assert_frame_assignment(self.frames_oa850, new_frames)

        self._masked_frames_oa850 = None
        self._vessel_masked_frames_oa850 = None
        self._fully_masked_frames_oa850 = None

        self._registered_frames_oa850 = None
        self._registered_mask_frames_oa850 = None
        self._registered_vessel_mask_oa850 = None

        self._frames_oa850 = new_frames

    @property
    def registered_frames_oa850(self):
        from image_processing import ImageRegistrator
        if self._registered_frames_oa850 is None:
            ir: ImageRegistrator = self.image_registrator
            self._registered_frames_oa850 = np.empty_like(self.frames_oa850)
            for i, frame in enumerate(self.frames_oa850):
                self._registered_frames_oa850[i] = ir.apply_registration(frame)

        return self._registered_frames_oa850

    @property
    def registered_mask_frames_oa850(self):
        from image_processing import ImageRegistrator
        if self._registered_mask_frames_oa850 is None:
            self._registered_mask_frames_oa850 = np.empty_like(self.mask_frames_oa850)
            ir: ImageRegistrator = self.image_registrator
            for i, mask in enumerate(self.mask_frames_oa850):
                self._registered_mask_frames_oa850[i] = ir.apply_registration(mask)
        return self._registered_mask_frames_oa850

    @property
    def registered_vessel_mask_oa850(self):
        from image_processing import ImageRegistrator
        if self._registered_vessel_mask_oa850 is None:
            ir: ImageRegistrator = self.image_registrator
            self._registered_vessel_mask_oa850 = ir.apply_registration(self.vessel_mask_oa850).astype(np.bool8)
        return self._registered_vessel_mask_oa850

    @property
    def image_registrator(self) -> image_processing.ImageRegistrator:
        from image_processing import ImageRegistrator
        if self._image_registrator is None:
            self._image_registrator = ImageRegistrator(source=self.vessel_mask_oa850, target=self.vessel_mask_oa790)
            self._image_registrator.register_vertically()
        return self._image_registrator

    def visualize_registration(self, figsize=(120, 150), fontsize=50,
                               linewidth=15, linestyle='--',
                               **supbplots_kwargs, ):
        from plotutils import no_ticks
        import matplotlib.lines

        fig, axes = plt.subplots(1, 4, figsize=figsize, **supbplots_kwargs)
        plt.rcParams['axes.titlesize'] = 135
        plt.subplots_adjust(top=50)
        no_ticks(axes)

        overlay_mask = np.zeros((*self.vessel_mask_oa790.shape, 3))
        overlay_mask[..., 0] = self.vessel_mask_oa790

        axes[0].imshow(overlay_mask)
        axes[0].set_title('Vessel mask oa790', pad=25, fontsize=fontsize)

        overlay_mask[..., 0] = 0
        overlay_mask[..., 1] = self.vessel_mask_oa850
        axes[1].imshow(overlay_mask)
        axes[1].set_title('Vessel mask oa850', pad=25, fontsize=fontsize)

        overlay_mask[..., 1] = self.registered_vessel_mask_oa850
        axes[2].imshow(overlay_mask)
        axes[2].set_title('Registered vessel mask oa850', pad=25, fontsize=fontsize)

        vessel_mask_oa790 = self.vessel_mask_oa790.copy()
        vessel_mask_oa790[:self.image_registrator.vertical_displacement, :] = 0
        overlay_mask[..., 0] = vessel_mask_oa790
        overlay_mask[..., 1] = self.registered_vessel_mask_oa850

        from evaluation import dice
        axes[3].imshow(overlay_mask)
        axes[3].set_title(f'Mask overlap. Dice {dice(vessel_mask_oa790, self.registered_vessel_mask_oa850):.3f}',
                          pad=25, fontsize=fontsize)


        plt.tight_layout()
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()

        coord1 = transFigure.transform(axes[0].transData.transform([0, self.image_registrator.vertical_displacement]))
        coord2 = transFigure.transform(
            axes[2].transData.transform([self.vessel_mask_oa850.shape[-1], self.image_registrator.vertical_displacement]))

        line = matplotlib.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                       transform=fig.transFigure, linewidth=linewidth, linestyle=linestyle)
        fig.lines.append(line)
        return fig

    @property
    def frames_confocal(self):
        if self._frames_confocal is None:
            self._frames_confocal = get_frames_from_video(self.video_confocal_file)[..., 0]
        return self._frames_confocal

    @frames_confocal.setter
    def frames_confocal(self, new_frames):
        VideoSession._assert_frame_assignment(self.frames_confocal, new_frames)
        self._masked_frames_confocal = None
        self._vessel_masked_frames_confocal = None
        self._fully_masked_frames_confocal = None
        self._frames_confocal = new_frames

    @staticmethod
    def _rectify_mask_frames(masks, crop_left=0):
        """ Rectify masks so that they are square.

        The original masks shape is irregular which can cause some inconveniences and problems.
        Rectify masks by clipping the irregular borders to straight lines so that the final masks
        is a rectangle.
        """
        import cv2
        cropped_masks = np.zeros_like(masks)

        for i, mask in enumerate(masks):
            # add a small border around the masks to detect changes on that are on the border
            mask_padded = cv2.copyMakeBorder(np.uint8(mask), 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)

            # Find where pixels go from black to white and from white to black (searching left to right)
            ys, xs = np.where(np.diff(mask_padded, axis=-1))

            # xs smaller than mean are on the left side and xs bigger than mean are on the right
            left_xs = xs[xs < xs.mean()]
            right_xs = xs[xs > xs.mean()]

            # clip masks to make it have straight lines
            m = np.bool8(mask_padded)
            m[:, :left_xs.max() + crop_left] = False
            m[:, right_xs.min():] = 0

            # remove the borders
            m = np.bool8(m[1:-1, 1:-1])

            cropped_masks[i, ...] = m

        assert cropped_masks.shape == masks.shape
        return cropped_masks

    @property
    def mask_frames_oa790(self):
        if self._mask_frames_oa790 is None:
            if self.mask_video_oa790_file == '':
                self.mask_frames_oa790 = np.ones_like(self.frames_oa790, dtype=np.bool8)
            else:
                self.mask_frames_oa790 = VideoSession._rectify_mask_frames(
                    get_frames_from_video(self.mask_video_oa790_file)[..., 0].astype(np.bool8), crop_left=15)

        return self._mask_frames_oa790

    @mask_frames_oa790.setter
    def mask_frames_oa790(self, masks):
        assert masks.dtype == np.bool8, f'The masks type must be {np.bool8}'
        assert masks.shape == self.frames_oa790.shape, \
            f'The frame masks must have the same shape as the frames. ' \
            f'frames oa790 shape:{self.frames_oa790.shape}, masks given shape:{masks.shape}'
        self._mask_frames_oa790 = VideoSession._rectify_mask_frames(masks)
        self._masked_frames_oa790 = None
        self._fully_masked_frames_oa790 = None

    @property
    def mask_frames_oa850(self):
        if self._mask_frames_oa850 is None:
            if self.mask_video_oa850_file == '':
                self.mask_frames_oa850 = np.ones_like(self.frames_oa850, dtype=np.bool8)
            else:
                self.mask_frames_oa850 = VideoSession._rectify_mask_frames(
                    get_frames_from_video(self.mask_video_oa850_file)[..., 0].astype(np.bool8), crop_left=15)

        return self._mask_frames_oa850

    @mask_frames_oa850.setter
    def mask_frames_oa850(self, masks):
        assert masks.dtype == np.bool8, f'The masks type must be {np.bool8}'
        assert masks.shape == self.frames_oa850.shape, \
            f'The frame masks must have the same shape as the frames. ' \
            f'frames oa850 shape:{self.frames_oa850.shape}, masks given shape:{masks.shape}'
        self._mask_frames_oa850 = VideoSession._rectify_mask_frames(masks)
        self._masked_frames_oa850 = None
        self._fully_masked_frames_oa850 = None

    @property
    def mask_frames_confocal(self):
        if self._mask_frames_confocal is None:
            if self.mask_video_confocal_file == '':
                self.mask_frames_confocal = np.ones_like(self.frames_confocal, dtype=np.bool8)
            else:
                self.mask_frames_confocal = VideoSession._rectify_mask_frames(
                    get_frames_from_video(self.mask_video_confocal_file)[..., 0].astype(
                        np.bool8), crop_left=15)

        return self._mask_frames_confocal

    @mask_frames_confocal.setter
    def mask_frames_confocal(self, masks):
        assert masks.dtype == np.bool8, f'The masks type must be {np.bool8}'
        assert masks.shape == self.frames_confocal.shape, \
            f'The frame masks must have the same shape as the frames. ' \
            f'frames confocal shape:{self.frames_confocal.shape}, masks given shape:{masks.shape}'
        self._mask_frames_confocal = VideoSession._rectify_mask_frames(masks)
        self._masked_frames_confocal = None
        self._fully_masked_frames_confocal = None

    @property
    def masked_frames_oa790(self):
        """ The frames from the oa790nm channel masked with the corresponding frames of the masked video.
        """
        if self._masked_frames_oa790 is None:
            # We invert the masks because True values mean that the values are masked and therefor invalid.
            # see: https://numpy.org/doc/stable/reference/maskedarray.generic.html
            self._masked_frames_oa790 = np.ma.masked_array(self.frames_oa790,
                                                           ~self.mask_frames_oa790)
        return self._masked_frames_oa790

    @property
    def masked_frames_oa850(self):
        """ The frames from the oa850nm channel masked with the corresponding frames of the masked video.
        """
        if self._masked_frames_oa850 is None:
            # We invert the masks because True values mean that the values are masked and therefor invalid.
            # see: https://numpy.org/doc/stable/reference/maskedarray.generic.html
            self._masked_frames_oa850 = np.ma.masked_array(self.frames_oa850,
                                                           ~self.mask_frames_oa850[:len(self.frames_oa850)])
        return self._masked_frames_oa850

    @property
    def masked_frames_confocal(self):
        """ The frames from the confocal channel masked with the corresponding frames of the masked video.
        """
        if self._masked_frames_confocal is None:
            # We invert the masks because True values mean that the values are masked and therefor invalid.
            # see: https://numpy.org/doc/stable/reference/maskedarray.generic.html
            self._masked_frames_confocal = np.ma.masked_array(self.frames_confocal,
                                                              ~self.mask_frames_confocal[:len(self.frames_oa850)])
        return self._masked_frames_confocal

    @property
    def vessel_masked_frames_oa790(self):
        """ The frames from the oa790nm channel masked with the vessel masks image.
        """
        if self._vessel_masked_frames_oa790 is None:
            self._vessel_masked_frames_oa790 = np.ma.empty_like(self.frames_oa790)
            for i, frame in enumerate(self.frames_oa790):
                self._vessel_masked_frames_oa790[i] = np.ma.masked_array(frame, ~self.vessel_mask_confocal)
        return self._vessel_masked_frames_oa790

    @property
    def vessel_masked_frames_oa850(self):
        """ The frames from the oa850nm channel masked with the vessel masks image.
        """
        if self._vessel_masked_frames_oa850 is None:
            self._vessel_masked_frames_oa850 = np.ma.empty_like(self.frames_oa850)
            for i, frame in enumerate(self.frames_oa850):
                self._vessel_masked_frames_oa850[i] = np.ma.masked_array(frame, ~self.vessel_mask_oa850)
        return self._vessel_masked_frames_oa850

    @property
    def vessel_masked_frames_confocal(self):
        """ The frames from the confocal channel masked with the vessel masks image.
        """
        if self._vessel_masked_frames_confocal is None:
            self._vessel_masked_frames_confocal = np.ma.empty_like(self.frames_confocal)
            for i, frame in enumerate(self.frames_confocal):
                self._vessel_masked_frames_confocal[i] = np.ma.masked_array(frame, ~self.vessel_mask_confocal)
        return self._vessel_masked_frames_confocal

    @property
    def fully_masked_frames_oa790(self):
        """ The frames from the oa790nm channel masked with the vessel masks image and the masks frames from the masks video.
        """
        if self._fully_masked_frames_oa790 is None:
            self._fully_masked_frames_oa790 = np.ma.empty_like(self.frames_oa790)
            for i, frame in enumerate(self.frames_oa790):
                self._fully_masked_frames_oa790[i] = np.ma.masked_array(frame, ~(
                        self.vessel_mask_oa790 * self.mask_frames_oa790[i]))
        return self._fully_masked_frames_oa790

    @property
    def fully_masked_frames_oa850(self):
        """ The frames from the oa850nm channel masked with the vessel masks image and the masks frames from the masks video.
        """
        if self._fully_masked_frames_oa850 is None:
            self._fully_masked_frames_oa850 = np.ma.empty_like(self.frames_oa850)
            for i, frame in enumerate(self.frames_oa850):
                self._fully_masked_frames_oa850[i] = np.ma.masked_array(frame, ~(
                        self.vessel_mask_oa850 * self.masked_frames_oa850[i]))
        return self._fully_masked_frames_oa850

    @property
    def fully_masked_frames_confocal(self):
        """ The frames from the confocalnm channel masked with the vessel masks image and the masks frames from the masks video.
        """
        if self._fully_masked_frames_confocal is None:
            self._fully_masked_frames_confocal = np.ma.empty_like(self.frames_confocal)
            for i, frame in enumerate(self.frames_confocal):
                self._fully_masked_frames_confocal[i] = np.ma.masked_array(frame, ~(
                        self.vessel_mask_confocal * self.mask_frames_confocal[i]))
        return self._fully_masked_frames_confocal

    @property
    def marked_frames_oa790(self):
        if self._marked_frames_oa790 is None:
            if not self.has_marked_video:
                raise Exception(f"Video session '{basename(self.video_oa790_file)}' has no marked video.")
            self._marked_frames_oa790 = get_frames_from_video(self.marked_video_oa790_file)[..., 0]
        return self._marked_frames_oa790

    @property
    def marked_frames_oa850(self):
        if self._marked_frames_oa850 is None:
            if self.marked_video_oa850_file == '':
                raise Exception(f"Video session '{basename(self.video_oa790_file)}' has no oa850 marked video.")
            self._marked_frames_oa850 = get_frames_from_video(self.marked_video_oa850_file)[..., 0]
        return self._marked_frames_oa850

    @property
    def cell_position_csv_files(self):
        """ Immutable list of filenames of the csvs with the cell points.
        """
        return self._cell_position_csv_files.copy()

    def _add_to_cell_positions(self, csv_file):
        """ Warning, assumes that each csv_file has unique indices and overwrites entries from those indices
        without checking the actual coordinates.
        """
        csv_cell_positions_df = pd.read_csv(csv_file, delimiter=',')

        csv_cell_positions_coordinates = np.int32(csv_cell_positions_df[['X', 'Y']].to_numpy())
        csv_cell_positions_frame_indices = np.int32(csv_cell_positions_df[['Slice']].to_numpy())

        # The csv file is 1 indexed but python is 0 indexed so we -1.
        frame_indices_all = np.int32(np.squeeze(csv_cell_positions_frame_indices - 1))
        frame_indices_unique = np.unique(frame_indices_all)

        # Number of cells in videos is the same as the number of entries in the csv_file
        for frame_idx in frame_indices_unique:
            curr_coordinates = csv_cell_positions_coordinates[
                np.where(frame_indices_all == frame_idx)[0]
            ]
            if frame_idx in self._cell_positions:
                warnings.warn(f"Same slice index, '{frame_idx + 1}', found in multiple csv_files."
                              f' Overwriting with the latest csv file coordinates.')

            # warning overwriting coordinates in  frame_idx if already exist
            self._cell_positions[frame_idx] = curr_coordinates
        self._cell_positions = OrderedDict(sorted(self._cell_positions.items()))

    @property
    def validation_frame_idx(self):
        if self._validation_frame_idx is None:
            max_positions = 0
            max_positions_frame_idx = list(self.cell_positions.keys())[0]

            for frame_idx in self.cell_positions:
                # find and assign the frame with the most cell points as a validation frame.

                # We don't want frame index to be the first frame for the usual case of temporal width 1.
                # We also  want to have some distance from the last (in case of motion contrast enhanced frames)
                if frame_idx == 0 or frame_idx >= len(self.frames_oa790) - 3:
                    continue

                cur_coordinates = self.cell_positions[frame_idx]

                if len(cur_coordinates) > max_positions and frame_idx != 0 and frame_idx != len(self.frames_oa790) - 2:
                    max_positions = len(cur_coordinates)
                    max_positions_frame_idx = frame_idx

            self._validation_frame_idx = max_positions_frame_idx

        return self._validation_frame_idx

    @validation_frame_idx.setter
    def validation_frame_idx(self, idx):
        assert idx in self.cell_positions, f'Frame index {idx} is not marked. Please assign a marked frame frame idx for validation'
        self._validation_frame_idx = idx

    def _remove_cell_positions(self, csv_file):
        """ Warning, assumes that each csv_file has unique indices and removes entries from those indices
        without checking the actual coordinates.
        """

        csv_cell_positions_df = pd.read_csv(csv_file, delimiter=',')

        # csv_cell_positions_coordinates = np.int32(csv_cell_positions_df[['X', 'Y']].to_numpy())
        csv_cell_positions_frame_indices = np.int32(csv_cell_positions_df[['Slice']].to_numpy())

        # The csv file is 1 indexed but python is 0 indexed so we -1.
        frame_indices = np.int32(np.unique(csv_cell_positions_frame_indices) - 1)

        for frame_idx in frame_indices:
            del self._cell_positions[frame_idx]

    def _initialise_cell_positions(self):
        for csv_file in self._cell_position_csv_files:
            self._add_to_cell_positions(csv_file)

    def append_cell_position_csv_file(self, csv_file):
        """ Adds cell points from the csv file
        """
        self._cell_position_csv_files.append(csv_file)
        if len(self._cell_positions) == 0:
            self._initialise_cell_positions()
        else:
            self._add_to_cell_positions(csv_file)

    def pop_cell_position_csv_file(self, idx):
        """ Remove the csv cell points from the csv file at index idx
        """
        self._cell_position_csv_files.pop(idx)
        if len(self._cell_positions) == 0:
            self._initialise_cell_positions()
        else:
            pass

    def remove_cell_position_csv_file(self, csv_file):
        """ Remove the csv cell points from the csv file at index idx
        """
        self._cell_position_csv_files.remove(csv_file)
        if len(self._cell_positions) == 0:
            self._initialise_cell_positions()
        else:
            pass

    @property
    def cell_positions(self):
        """ A dictionary with {frame index -> Nx2 x,y cell points}.

        Returns the points of the blood cells as a dictionary indexed by the frame index as is in the csv file
        but 0 indexed instead!. To get the first frame do  session.points[0] instead of session.points[1].

        To access ith frame's cell points do:
        self.points[i - 1]
        """
        if len(self._cell_position_csv_files) == 0:
            raise Exception(f"No csv found with cell points for video session {basename(self.video_oa790_file)}")

        if len(self._cell_positions) == 0:
            self._initialise_cell_positions()

        return self._cell_positions

    @staticmethod
    def _vessel_mask_from_file(file):
        vessel_mask = plt.imread(file)
        if len(vessel_mask.shape) == 3:
            vessel_mask = vessel_mask[..., 0]

        return np.bool8(vessel_mask)

    def load_vessel_masks(self, v=False):
        try:
            self.vessel_mask_oa790 = VideoSession._vessel_mask_from_file(self.vessel_mask_oa790_file)
            if v:
                print('Loaded', self.vessel_mask_oa790_file)
        except:
            if v:
                print('No vessel mask for oa790nm channel found')

        try:
            self.vessel_mask_oa850 = VideoSession._vessel_mask_from_file(self.vessel_mask_oa850_file)
            if v:
                print('Loaded', self.vessel_mask_oa850_file)
        except:
            if v:
                print('No vessel mask for oa850nm channel found')

        try:
            self.vessel_mask_confocal = VideoSession._vessel_mask_from_file(self.vessel_mask_confocal_file)
            if v:
                print('Loaded', self.vessel_mask_confocal_file)
        except:
            if v:
                print('No vessel mask for confocal video found')

    @property
    def vessel_mask_oa790(self):
        from vessel_detection import create_vessel_mask_from_frames
        if self._vessel_mask_oa790 is None:
            if not self.vessel_mask_oa790_file or not self.load_vessel_mask_from_file:
                self._vessel_mask_oa790 = create_vessel_mask_from_frames(self.masked_frames_oa790)
            else:
                self._vessel_mask_oa790 = VideoSession._vessel_mask_from_file(self.vessel_mask_oa790_file)
        return self._vessel_mask_oa790

    @vessel_mask_oa790.setter
    def vessel_mask_oa790(self, val):
        if len(val.shape) == 3:
            val = val[..., 0]
        self._vessel_mask_oa790 = np.bool8(val)

        self._vessel_masked_frames_oa790 = None
        self._fully_masked_frames_oa790 = None

        self._image_registrator = None
        self._registered_frames_oa850 = None
        self._registered_mask_frames_oa850 = None
        self._registered_vessel_mask_oa850 = None

    @property
    def vessel_mask_oa850(self):
        from vessel_detection import create_vessel_mask_from_frames
        if self._vessel_mask_oa850 is None:
            if not self.vessel_mask_oa850_file or not self.load_vessel_mask_from_file:
                self._vessel_mask_oa850 = create_vessel_mask_from_frames(self.masked_frames_oa850)
            else:
                self._vessel_mask_oa850 = VideoSession._vessel_mask_from_file(self.vessel_mask_oa850_file)
        return self._vessel_mask_oa850

    @vessel_mask_oa850.setter
    def vessel_mask_oa850(self, val):
        if len(val.shape) == 3:
            val = val[..., 0]
        self._vessel_mask_oa850 = np.bool8(val)

        self._vessel_masked_frames_oa850 = None
        self._fully_masked_frames_oa850 = None

        self._image_registrator = None
        self._registered_frames_oa850 = None
        self._registered_mask_frames_oa850 = None
        self._registered_vessel_mask_oa850 = None

    @property
    def vessel_mask_confocal(self):
        from vessel_detection import create_vessel_mask_from_frames
        if self._vessel_mask_confocal is None:
            if not self.vessel_mask_confocal_file or not self.load_vessel_mask_from_file:
                self._vessel_mask_confocal = create_vessel_mask_from_frames(self.masked_frames_confocal)
            else:
                self._vessel_mask_confocal = VideoSession._vessel_mask_from_file(self.vessel_mask_confocal_file)
        return self._vessel_mask_confocal

    @vessel_mask_confocal.setter
    def vessel_mask_confocal(self, val):
        if len(val.shape) == 3:
            val = val[..., 0]

        self._vessel_mask_confocal = np.bool8(val)
        self._vessel_masked_frames_confocal = None
        self._fully_masked_frames_confocal = None

        self._registered_mask_frames_oa850 = None
        self._registered_mask_frames_oa850 = None

    @staticmethod
    def _std_image_from_file(file):
        from image_processing import normalize_data
        std_image = plt.imread(file)
        if len(std_image.shape) == 3:
            std_image = std_image[..., 0]

        # if image is 16 bit scale back to uint8 (assuming data range is from 0 to 2^16 -1)
        if std_image.dtype == np.uint16:
            uint16_max_val = np.iinfo(np.uint16).max
            std_image = np.uint8(normalize_data(std_image, target_range=(0, 255), data_range=(0, uint16_max_val)))

        return std_image

    @property
    def std_image_oa790(self):
        if self._std_image_oa790 is None:
            if self.std_image_oa790_file == '':
                raise Exception(f"No standard deviation image oa790 found for session '{self.video_oa790_file}'")
            self._std_image_oa790 = VideoSession._std_image_from_file(self.std_image_oa790_file)
        return self._std_image_oa790

    @property
    def std_image_oa850(self):
        if self._std_image_oa850 is None:
            if self.std_image_oa850_file == '':
                raise Exception(f"No standard deviation image oa850 found for session '{self.video_oa790_file}'")
            self._std_image_oa850 = VideoSession._std_image_from_file(self.std_image_oa850_file)
        return self._std_image_oa850

    @property
    def std_image_confocal(self):
        if self._std_image_confocal is None:
            if self.std_image_confocal_file == '':
                raise Exception(f"No standard deviation image confocal found for session '{self.video_oa790_file}'")
            self._std_image_confocal = VideoSession._std_image_from_file(self.std_image_confocal_file)
        return self._std_image_confocal


def get_video_sessions(
        marked=True,
        registered=True,
        validation=False,
        load_vessel_mask_from_file=True,
        v=False,
):
    """ Get video sessions

    Please see VideoSession.

    Args:
        marked (bool): If True returns only video sessions that have a csv with the cell locations
        registered (bool): If True returns only video sessions that have videos that are registered
        validation (bool): If True returns videos that should be used for validation
        load_vessel_mask_from_file (bool):
         If True loads the vessel masks from an existing video if it exists to save time on calculating the vessel mask
         for the training, validation and confocal videos.
        v (bool): If true have verbose output

    Returns:
        List[VideoSession] List of video sessions
    """
    from shared_variables import unmarked_video_oa790_filenames
    video_sessions: List[VideoSession] = []

    uids = []
    for video_filename in unmarked_video_oa790_filenames:
        vs = VideoSession(video_filename, load_vessel_mask_from_file=load_vessel_mask_from_file)
        if load_vessel_mask_from_file:
            vs.load_vessel_masks(v=v)

        # every video session has a unique uid, make sure only one object created per unique video session
        if vs.uid in uids:
            continue

        if marked and not vs.is_marked:
            continue
        elif registered and not vs.is_registered:
            continue
        elif validation and not vs.is_validation:
            continue
        elif not validation and vs.is_validation:
            continue

        video_sessions.append(vs)

    return video_sessions


class SessionPreprocessor(object):
    _preprocess_functions: List[Any]

    def __init__(self, session: VideoSession, preprocess_functions=None):
        if preprocess_functions is None:
            preprocess_functions = []

        self._preprocess_functions = preprocess_functions
        self._session = session

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, vs: VideoSession):
        self._session = vs

    def reset_preprocess(self):
        self._preprocess_functions = []

    def with_session(self, vs: VideoSession):
        self.session = vs
        return self

    def with_preprocess(self, preprocess_functions):
        try:
            self._preprocess_functions.extend(preprocess_functions)
        except TypeError:
            # if not a list just append
            self._preprocess_functions.append(preprocess_functions)
        return self

    def map(self):
        for preprocessing_func in self._preprocess_functions:
            self.session.frames_oa790 = np.array(list(map(preprocessing_func, self.session.frames_oa790)))

    def _apply_preprocessing(self, masked_frames):
        masked_frames = masked_frames.copy()
        print(self._preprocess_functions)
        for fun in self._preprocess_functions:
            masked_frames = fun(masked_frames)
        return masked_frames

    def apply_preprocessing_to_oa790(self):
        masked_frames = self._apply_preprocessing(self.session.masked_frames_oa790)
        if not np.ma.is_masked(masked_frames):
            masked_frames = np.ma.masked_array(masked_frames, self.session.masked_frames_oa790.mask)
        self.session.frames_oa790 = masked_frames.filled(masked_frames.mean())
        self.session.mask_frames_oa790 = ~masked_frames.mask

    def apply_preprocessing_to_oa850(self):
        print('how')
        masked_frames = self._apply_preprocessing(self.session.masked_frames_oa850)
        if not np.ma.is_masked(masked_frames):
            masked_frames = np.ma.masked_array(masked_frames, self.session.masked_frames_oa850.mask)
        self.session.frames_oa850 = masked_frames.filled(masked_frames.mean())
        self.session.mask_frames_oa850 = ~masked_frames.mask

    def apply_preprocessing_to_confocal(self):
        print('what')
        masked_frames = self._apply_preprocessing(self.session.masked_frames_confocal)
        if not np.ma.is_masked(masked_frames):
            masked_frames = np.ma.masked_array(masked_frames, self.session.masked_frames_confocal.mask)
        self.session.frames_confocal = masked_frames.filled(masked_frames.mean())
        self.session.mask_frames_confocal = masked_frames.mask

    def apply_preprocessing(self):
        self.apply_preprocessing_to_confocal()
        self.apply_preprocessing_to_oa790()
        self.apply_preprocessing_to_oa850()
