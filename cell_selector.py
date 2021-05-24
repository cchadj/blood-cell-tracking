"""
python estimated_location_keeper -c estimated_coords.csv -v video.avi -o output.csv
"""
import os
from os.path import basename
from typing import Dict

from shared_variables import OUTPUT_FOLDER
import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from guitools import MplScatterPlotPointSelector
import pathlib
from matplotlib.widgets import Button
from matplotlib import widgets
import tkinter as tk
from tkinter import filedialog
from video_session import VideoSession

# thanks to https://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel
def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    return zoom_fun


class MplFrameSelector(object):
    fig: plt.Figure
    point_selector_dict: Dict[int, MplScatterPlotPointSelector]

    def __init__(self,
                 frames,
                 cell_coordinates,
                 output_file,
                 frame_masks=None,
                 vessel_mask=None):
        """

        Args:
            frames: NxHxWxC, or NxHxW
            frame_indices:
            cell_coordinates: dict, where points[frame_index] = Nx2. The cell points for each frame.
        """
        self.frames = frames
        self.frame_masks = frame_masks
        self.vessel_mask = vessel_mask
        self.cell_cords_dict = cell_coordinates
        self.output_file = output_file

        self.key_idx = 0

        self._frame_idx = 0
        self.cur_frame_cell_cords = None
        self.cur_frame = None
        self.cur_mask = None
        self.cur_point_selector = None
        # Holds a point selector object for each frame_idx
        self.point_selector_dict = {}
        self.fig, self.ax = None, None

    def new_save(self, *args, **kwargs):
        print("save_figure pressed")
        return "break"

    def activate(self):
        plt.rcParams['toolbar'] = 'None'
        scale = 1.5
        self.fig, self.ax = plt.subplots()
        plt.tight_layout()
        # keep reference to avoid garbage collection
        # noinspection PyAttributeOutsideInit
        self.f = zoom_factory(self.ax, base_scale=scale)
        if self.frame_idx is None:
            self.frame_idx = list(self.cell_cords_dict.keys())[0]
        self.update()

    def deactivate(self):
        for frame_idx in self.point_selector_dict:
            point_selector = self.point_selector_dict[frame_idx]
            if point_selector.is_activated:
                point_selector.deactivate()

    def close(self):
        self.deactivate()
        plt.close(self.fig)

    @property
    def frame_idx(self):
        return int(self._frame_idx)

    @frame_idx.setter
    def frame_idx(self, val):
        self._frame_idx = min(val, len(self.frames) - 1)
        self.update()

    @classmethod
    def fromvideosession(cls, video_session: VideoSession):
        from datetime import datetime
        """ Create cell selector from VideoSession object

            video_session (VideoSession):
        """

        filename_without_extension = basename(video_session.video_oa790_file).split('.')[0]
        output_file = os.path.join(OUTPUT_FOLDER,
                                   f'{datetime.now().strftime("date_time%m-%d-%Y_%H-%M-%S")}',
                                   filename_without_extension + '_selected_cords.csv')
        try:
            frame_masks = video_session.mask_frames_oa790
        except Exception:
            frame_masks = None

        try:
            vessel_mask = video_session.vessel_mask_oa790
        except Exception:
            try:
                vessel_mask = video_session.vessel_mask_confocal
            except Exception:
                vessel_mask = None

        return cls(video_session.frames_oa790,
                   video_session.cell_positions,
                   output_file, frame_masks, vessel_mask)

    def prev_marked_frame(self):
        marked_frame_indices = np.array(list(self.cell_cords_dict.keys()))
        # keep only those that are bigger than cur frame idx
        marked_frame_indices = marked_frame_indices[marked_frame_indices < self._frame_idx]

        if len(marked_frame_indices) == 0:
            # if only the current frame idx is smaller or same than any other marked frame index go to last marked frame
            self.frame_idx = list(self.cell_cords_dict.keys())[-1]
        else:
            # go to closest next frame
            self.frame_idx = marked_frame_indices[np.argmin(np.abs(marked_frame_indices - self._frame_idx))]

    def next_marked_frame(self):
        marked_frame_indices = np.array(list(self.cell_cords_dict.keys()))
        # keep only those that are bigger than cur frame idx
        marked_frame_indices = marked_frame_indices[marked_frame_indices > self._frame_idx]

        if len(marked_frame_indices) == 0:
            # if only the current frame idx is bigger or same than any other marked frame index go to first marked frame
            self.frame_idx = list(self.cell_cords_dict.keys())[0]
        else:
            # go to closest next frame
            self.frame_idx = marked_frame_indices[np.argmin(np.abs(marked_frame_indices - self._frame_idx))]

    def prev_frame(self):
        self._frame_idx = (self._frame_idx - 1) % len(self.frames)
        self.update()

    def update(self):
        """ Update frame selector plot contents

        Args:
            frame_idx:

        Returns:

        """
        # Get frame at index and masks at that frame.
        self.cur_frame = self.frames[self.frame_idx]

        if self.frame_idx in self.cell_cords_dict:
            self.cur_frame_cell_cords = self.cell_cords_dict[self.frame_idx]
        else:
            self.cur_frame_cell_cords = None
        if self.frame_idx not in self.point_selector_dict and self.cur_frame_cell_cords is not None:
            self.point_selector_dict[self.frame_idx] = MplScatterPlotPointSelector(self.cur_frame_cell_cords, fig_ax=(self.fig, self.ax))

        # Update cur point selector. Each frame has it's own point selector.
        if self.cur_point_selector is not None:
            self.cur_point_selector.deactivate()
        if self.frame_idx in self.point_selector_dict:
            self.cur_point_selector = self.point_selector_dict[self.frame_idx]
        else:
            self.cur_point_selector = None

        # apply masks from the frame masks video.
        if self.frame_masks is not None:
            self.cur_mask = self.frame_masks[self.frame_idx]
            self.cur_frame[~self.cur_mask] = 0

        # apply masks from the vessel masks
        if self.vessel_mask is not None:
            self.cur_frame[~self.vessel_mask] = 0

        self.ax.clear()
        self.ax.imshow(self.cur_frame, cmap='gray')

        # self.ax.scatter(self.cur_coords[:, 0], self.cur_coords[:, 1])
        self.ax.set_title(f'Frame {self.frame_idx}')
        if self.cur_point_selector is not None:
            self.cur_point_selector.activate()

        plt.ioff()
        self.fig.canvas.draw_idle()
        plt.ioff()

    def save(self):
        output_df = pd.DataFrame(columns=['X', 'Y', 'Slice'])
        for frame_idx in self.cell_cords_dict.keys():
            if frame_idx not in self.point_selector_dict:
                continue
            point_selector = self.point_selector_dict[frame_idx]
            if len(point_selector.selected_point_indices) == 0:
                # when no point selected from that frame
                continue

            positions_to_keep = point_selector.all_points[point_selector.selected_point_indices, :]

            # the frame indices in the csv are 1 indexed so we add 1
            slice_idx = np.ones(len(positions_to_keep), dtype=np.int) * (frame_idx + 1)
            cur_coords_df = pd.DataFrame({
                'X': positions_to_keep[:, 0],
                'Y': positions_to_keep[:, 1],
                'Slice': slice_idx})
            output_df = output_df.append(cur_coords_df)

        pathlib.Path(os.path.dirname(self.output_file)).mkdir(parents=True, exist_ok=True)
        output_df.to_csv(self.output_file)
        print(f'Saved output to {self.output_file}')


def dir_path(path):
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        return path
    except:
        raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')


def parse_arguments():
    """ Parse the arguments and get the video filename and the coordinates along with the frame indices
    (starting from 1)

    The returned coordinates and frame indices should have the same length.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=argparse.FileType('r'), required=False, nargs=1,
                        help='The video that the estimated locations belong to.')
    parser.add_argument('-m', '--masks-video', type=argparse.FileType('r'), required=False, nargs='*',
                        help='The video with the corresponding rectangle masks.'
                             ' Must be of same length as the retinal video. Can be omitted.'
                             ' If set but no argument given then file picker window is opened.')
    parser.add_argument('--vessel-masks', type=argparse.FileType('r'), required=False, nargs='*',
                        help='Mask that highlights the vessels. '
                             'Must be the same size as the frames. Can be omitted.'
                             'If set but no argument given then a file picker window is opened.')
    parser.add_argument('-c', '--coords', type=argparse.FileType('r'), required=False, nargs=1,
                        help='The coordinates csv. Must have 3 columns, X, Y, Slice. '
                             'X, Y is the location of the coordinate and Slice is the frame starting from 1')
    parser.add_argument('-o', '--output-directory', type=dir_path, default='.',
                        help='The directory for the output file. The created will have the same name as the video'
                             "with '_selected_coords.csv' appended to it.")
    args = parser.parse_args()

    # Get the retinal video filename
    video_filename = ""
    try:
        video_filename = args.video[0].name
    except:
        root = tk.Tk()
        root.withdraw()
        print('Select video.')
        while video_filename == "":
            video_filename = tk.filedialog.askopenfilename(title='Select retinal video.',
                                                       filetypes=[('Video files', ['*.avi', '*.flv', '*.mov', '*.mp4',
                                                                                   '*.wmv', '*.qt', '*.mkv'])])
            if video_filename == "":
                print('The retinal video is required. Please select video.')

    # Get the csv with the bloodcell points filename
    csv_filename = ""
    try:
        csv_filename = args.coords[0].name
    except:
        root = tk.Tk()
        root.withdraw()
        print('Select csv file with blood sell points.')
        while csv_filename == "":
            csv_filename = tk.filedialog.askopenfilename(title='Select csv file with blood cell points',
                                                         filetypes=[('CSV files', ['*.txt', '*.csv'])])
            if csv_filename == "":
                print('The csv file with the bloodcell points is required. Please select csv.')

    coordinates_df = pd.read_csv(csv_filename, sep=',')
    cell_positions = coordinates_df[['X', 'Y']].to_numpy()
    frame_indices = coordinates_df['Slice'].to_numpy()

    # Optional video with the rectangle masks for each frame filename.
    masks_video_filename = None
    if 'masks_video_filename' in args:
        try:
            masks_video_filename = args.mask_video[0].name
        except:
            root = tk.Tk()
            root.withdraw()
            # --masks-video set but no argument given. Open file picker window.
            print('Select masks video.')
            masks_video_filename = tk.filedialog.askopenfilename(title='Select video masks',
                                                                 filetypes=[('Video files', ['*.avi', '*.flv', '*.mov', '*.mp4',
                                                                                             '*.wmv', '*.qt', '*.mkv'])])

    # Optional image with the vessel masks to isolate capillaries.
    vessel_mask = None
    if 'vessel_mask' in args:
        try:
            vessel_mask_filename = args.vessel_mask[0].name
        except:
            root = tk.Tk()
            root.withdraw()
            # --vessel-masks set but no argument given. Open file picker window.
            vessel_mask_filename = tk.filedialog.askopenfilename(title='Select vessel masks.',
                                                                 filetypes=[('Image',  ['*.jpg', '*.tif', '*.png', '.bmp'
                                                                                        '*.jpeg', '*.tiff'])])

        vessel_mask = np.bool8(plt.imread(vessel_mask_filename)[..., 0])
    return video_filename, masks_video_filename, vessel_mask, cell_positions, frame_indices, args.output_directory


    video_filename, masks_video_filename, vessel_mask, coords, frame_indices, output_directory = parse_arguments()

    output_csv_filename = os.path.splitext(os.path.basename(video_filename))[0] + '_selected_coords.csv'
    output_csv_filename = os.path.join(output_directory, output_csv_filename)

    frames = get_frames_from_video(video_filename)

    # points[frame_idx] will contain 2xN_cells_in_frame array for each frame
    cell_positions = {}
    [frame_idxs, idxs] = np.unique(frame_indices, return_index=True)
    for i in range(len(frame_idxs)):
        curr_idx = idxs[i]
        frame_idx = frame_idxs[i]

        if i == len(frame_idxs) - 1:
            cell_positions[frame_idx] = (coords[curr_idx:-1])
        else:
            cell_positions[frame_idx] = coords[curr_idx:idxs[i + 1]]

    frame_masks = None
    if masks_video_filename:
        frame_masks = np.bool8(get_frames_from_video(masks_video_filename)[..., 0])
    callback = MplFrameSelector(frames, cell_positions, output_csv_filename,
                                frame_masks=frame_masks, vessel_mask=vessel_mask)

    # ax rect -> [left, bottom, width, height]
    axprev = plt.axes([0.85, 0.65, 0.1, 0.175])
    axtxtbox = plt.axes([0.85, 0.50, 0.1, 0.075])
    axnext = plt.axes([0.85, 0.25, 0.1, 0.175])
    axsave = plt.axes([0.85, 0.1, 0.1, .075])

    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next_frame)

    bsave = Button(axsave, 'Save')
    bsave.on_clicked(callback.save)

    frame_txtbox = widgets.TextBox(axtxtbox, label='Fr idx')
    frame_txtbox.on_submit(callback.select_frame)

    def ensure_number(text):
        try:
            if text is not "":
                n = int(text)
                if n >= len(frames):
                    frame_txtbox.set_val(callback._frame_idx)
        except ValueError:
            frame_txtbox.set_val(callback._frame_idx)

    frame_txtbox.on_text_change(ensure_number)

    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev_frame)
    plt.show()
