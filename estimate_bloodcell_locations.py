from PyQt5 import QtWidgets
from PyQt5.QtWidgets import  QApplication, QMainWindow

from shared_variables import *
import argparse
import pathlib
import re
from sys import exit
import numpy as np
import pandas as pd
import torch
from torch import nn
from classificationutils import create_probability_map, estimate_cell_positions_from_probability_map
from cnnlearning import CNN
import tqdm
from tqdm.contrib import tzip
from shared_variables import unmarked_video_oa790_filenames

DEFAULT_N_FRAMES_PER_VIDEO = 10


def dir_path(path):
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        return path
    except:
        raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')


def extract_value_from_string(string, value_prefix, delimiter='_'):
    strings = string.split(delimiter)
    val = None
    for i, s in enumerate(strings):
        if s == value_prefix:
            val = float(re.findall(r"[-+]?\d*\.\d+|\d+", strings[i + 1])[0])
            break

    return val


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=None, help='The classifier model. '
                                                            'If omitted then picks the one with the best validation'
                                                            'accuracy from the cache folder. '
                                                            'If no model is found in cache then this fails.')
    parser.add_argument('-v', '--videos', type=argparse.FileType('r'), nargs='*',
                        help='The video or videos to find the location of cells of. '
                             'If omitted then goes through the videos in the data folder.')
    parser.add_argument('-n', '--n-frames-per-video', type=int, default=None,
                        help='How many frames to process per frame.'
                             'This has priority over frame interval meaning if both'
                             '--frame-interval and --n-frames-per-video are set then only the latter takes place.'
                             f' Default is {DEFAULT_N_FRAMES_PER_VIDEO} frames.')
    parser.add_argument('-i', '--frame-interval', type=int, default=None,
                        help='Interval of frames for position estimation. '
                             'Has less priority than --n-frames-per-video meaning if'
                             'both are set then only -n is taken into consideration.')
    parser.add_argument('-o', '--output-directory', type=dir_path, default=OUTPUT_ESTIMATED_POSITIONS_FOLDER,
                        help='The output directory for the estimated position CSVs.')
    parser.add_argument('-p', '--patch-size', default=-1, type=int,
                        help='Patch size of cells. '
                             "If omitted then patch size is extracted from model file name, after '_ps_'. "
                             ' The patch size must be specified either in the model filename or with -p')

    args = parser.parse_args()

    if args.videos is None:
        video_filenames = unmarked_video_oa790_filenames
    else:
        video_filenames = [f.name for f in args.videos]

    model_filename = args.model
    if model_filename is None:
        max_valid_accuracy = 0
        try:
            files = os.listdir(CACHED_MODELS_FOLDER)
            if len(files) == 0:
                raise FileNotFoundError

            for f in files:
                valid_accuracy = float(extract_value_from_string(f, 'va'))
                if valid_accuracy > max_valid_accuracy:
                    max_valid_accuracy = valid_accuracy
                    best_model_filename = f
        except FileNotFoundError:
            print('No model provided and no models in cache')
            return -1
        model_filename = os.path.join(CACHED_MODELS_FOLDER, best_model_filename)
    model = CNN(convolutional=
    nn.Sequential(
        nn.Conv2d(1, 32, padding=2, kernel_size=5),
        # PrintLayer("1"),
        nn.BatchNorm2d(32),
        # PrintLayer("2"),
        nn.MaxPool2d(kernel_size=(3, 3), stride=2),
        # PrintLayer("3"),

        nn.Conv2d(32, 32, padding=2, kernel_size=5),
        # PrintLayer("4"),
        nn.BatchNorm2d(32),
        # PrintLayer("5"),
        nn.ReLU(),
        # PrintLayer("6"),
        nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        # PrintLayer("7"),

        nn.Conv2d(32, 64, padding=2, kernel_size=5),
        # PrintLayer("9"),
        nn.BatchNorm2d(64),
        # PrintLayer("11"),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        # PrintLayer("12"),
    ),
     dense=
        nn.Sequential(
            nn.Linear(576, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(64),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2),
            #   nn.Softmax()
        )).to(device)

    model.load_state_dict(torch.load(model_filename))
    model.eval()

    patch_size = extract_value_from_string(model_filename, 'ps')
    if patch_size is None:
        patch_size = args.patch_size
        if patch_size is None:
            print('Patch size not specified')
            return -1
    patch_size = int(patch_size)

    if args.n_frames_per_video is None and args.frame_interval is None:
        n_frames_per_video = DEFAULT_N_FRAMES_PER_VIDEO
    return video_filenames, model_filename, args.output_directory, model, args.frame_interval, n_frames_per_video, patch_size


def main():
    video_filenames, model_filename, output_directory, model, frame_interval, n_frames_per_video, patch_size = parse_arguments()

    for video in tqdm.tqdm(video_filenames):
        video_estimated_positions_df = pd.DataFrame(columns=['X', 'Y', 'Slice'])

        frames = get_frames_from_video(video, normalise=True)[..., 0]
        if n_frames_per_video is not None:
            frame_interval = int(len(frames) / n_frames_per_video)
        frame_indices = np.arange(len(frames), step=frame_interval)
        frames = frames[frame_indices, ...]
        # Slices in the csv files start from 1 instead of 0.
        frame_indices += 1

        print('\n--------------------------')
        print(f"Processing video: '{video}'")
        print(f'Processing {len(frames)} frames')

        output_csv_file = os.path.splitext(os.path.basename(video))[0] + '_estimated_coords.csv'
        output_csv_file = os.path.join(output_directory, output_csv_file)
        for frame, frame_idx in tzip(frames, frame_indices):
            probability_map = create_probability_map(frame, model, patch_size=patch_size)
            positions = estimate_cell_positions_from_probability_map(probability_map, sigma=1., extended_maxima_h=.45)
            positions_df = pd.DataFrame({'X': positions[:, 0],
                                         'Y': positions[:, 1],
                                         'Slice': np.ones(len(positions)) * frame_idx})
            video_estimated_positions_df = video_estimated_positions_df.append(positions_df, ignore_index=True)

        video_estimated_positions_df.to_csv(output_csv_file, sep=',')

        print(f"Estimated positions for saved as: '{output_csv_file}'")
    return 0


def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(0, 0, 300, 300)
    win.setWindowTitle('Example!')

    label = QtWidgets.QLabel(win)
    label.setText('This is an example')
    label.move(50, 50)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    window()
    exit(main())

