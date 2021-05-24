import os
import sys
from os.path import basename
import glob
from typing import List
import pathlib


def files_of_same_source(f1, f2):
    f1, f2 = os.path.basename(f1), os.path.basename(f2)
    f1_split = f1.split('_')
    f2_split = f2.split('_')

    return f1_split[5] == f2_split[5]


DATA_FOLDER = os.path.join('.', 'data')
CACHE_FOLDER = os.path.join('.', 'cache')
CACHED_MODELS_FOLDER = os.path.join(CACHE_FOLDER, 'models')
CACHED_DICE = os.path.join(CACHE_FOLDER, 'dice')
CACHED_DATASETS_FOLDER = os.path.join(CACHE_FOLDER, 'datasets')

OUTPUT_FOLDER = os.path.join('.', 'output')
OUTPUT_ESTIMATED_POSITIONS_FOLDER = os.path.join(OUTPUT_FOLDER, 'estimated-points')

pathlib.Path(OUTPUT_FOLDER).mkdir(exist_ok=True, parents=True)
pathlib.Path(OUTPUT_ESTIMATED_POSITIONS_FOLDER).mkdir(exist_ok=True, parents=True)

# Set up file extensions here. All extensions must be lowercase.
csv_file_extensions = ('.csv', '.txt')
video_file_extensions = ('.avi', '.webm', '.mp4')
image_file_extensions = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp')

# find all files in data folder
all_files = glob.glob(os.path.join(DATA_FOLDER, '**', '*.*'), recursive=True)

# # Remove duplicates based on the file basename.
all_files_basenames = [basename(f) for f in all_files]
seen = set()
all_files_uniq = []
all_files_basenames_uniq = []
for i, file in enumerate(all_files_basenames):
    if file not in seen:
        all_files_basenames_uniq.append(file)
        all_files_uniq.append(all_files[i])
        seen.add(file)
all_files = all_files_uniq
# sort based on basename
all_files = [f for _, f in sorted(zip(all_files_basenames_uniq, all_files))]
## end remove duplicates ##

all_video_files = [f for f in all_files if f.lower().endswith(video_file_extensions)]
all_csv_files = [f for f in all_files if f.lower().endswith(csv_file_extensions)]
all_image_files = [f for f in all_files if f.lower().endswith(image_file_extensions)]

# unmarked videos. Must not have '_marked' or '_mask' in them.
all_video_files_unmarked = [f for f in all_video_files if
                            '_marked' not in basename(f).lower() and '_mask' not in basename(f)]

unmarked_video_confocal_filenames = [f for f in all_video_files_unmarked if 'confocal' in basename(f).lower()]
unmarked_video_oa790_filenames = [f for f in all_video_files_unmarked if 'oa790' in basename(f).lower()]
unmarked_video_oa850_filenames = [f for f in all_video_files_unmarked if 'oa850' in basename(f).lower()]

# marked videos. Must not have 'mask'. Must have '_marked'.
all_video_files_marked = [f for f in all_video_files if '_marked' in basename(f).lower() and '_mask' not in basename(f)]
marked_video_oa790_files = [f for f in all_video_files_marked if 'oa790' in basename(f).lower()]
marked_video_oa850_files = [f for f in all_video_files_marked if 'oa850' in basename(f).lower()]

# mask videos. Must have '_mask.' in them.
all_mask_video_files = [f for f in all_video_files if '_mask.' in basename(f).lower()]
mask_video_oa790_files = [f for f in all_mask_video_files if 'oa790' in basename(f).lower()]
mask_video_oa850_files = [f for f in all_mask_video_files if 'oa850' in basename(f).lower()]
mask_video_confocal_files = [f for f in all_mask_video_files if 'confocal' in basename(f).lower().lower()]

# Csv files with blood-cell coordinate files.
all_csv_cell_cords_filenames = [f for f in all_csv_files if 'coords' in basename(f).lower() or 'cords' in basename(f)]
csv_cell_cords_oa790_filenames = [f for f in all_csv_cell_cords_filenames if 'oa790nm' in basename(f).lower()]
csv_cell_cords_oa850_filenames = [f for f in all_csv_cell_cords_filenames if 'oa850nm' in basename(f).lower()]

# mask images. must end with 'vessel_mask.<file_extension>'
all_vessel_mask_files = [f for f in all_image_files if '_vessel_mask.' in basename(f).lower()]
vessel_mask_oa790_files = [f for f in all_vessel_mask_files if 'oa790' in basename(f).lower()]
vessel_mask_oa850_files = [f for f in all_vessel_mask_files if 'oa850' in basename(f).lower()]
vessel_mask_confocal_files = [f for f in all_vessel_mask_files if 'confocal' in basename(f).lower()]

# standard deviation images. must end with '_std.<file_extension>
all_std_image_files = [f for f in all_image_files if '_std.' in basename(f).lower()]
std_image_oa790_files = [f for f in all_std_image_files if 'oa790' in basename(f).lower()]
std_image_oa850_files = [f for f in all_std_image_files if 'oa850' in basename(f).lower()]
std_image_confocal_files = [f for f in all_std_image_files if 'confocal' in basename(f).lower()]


def find_filename_of_same_source(target_filename, filenames):
    """ Find the file name in filenames that is of the same source of target filename.

    Args:
        target_filename:
        filenames:

    Returns:
        The filename in filenames that is of same source of target file name.
        If not found returns an empty string

    """
    for filename in filenames:
        if files_of_same_source(target_filename, filename):
            return filename
    return ''


if __name__ == '__main__':
    # Make sure that all_files is unique based on the basename of the file (being at different path doesn't matter)
    all_files_basenames_unique = list(set([basename(f) for f in all_files]))
    assert len(all_files_basenames_unique) == len(all_files) and \
           [basename(f1) == f2 for f1, f2 in zip(sorted(all_files), sorted(all_files_basenames_unique))], \
        'all_files is not unique, there is one or more duplicates.'

    sys.exit(0)
