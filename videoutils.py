import cv2
import numpy as np


def get_frames_from_video(video_filename, normalise=False):
    """
    Get the frames of a video as an array.

    Arguments:
        video_filename: The path of the video.
        normalise: Normalise frame values between 0 and 1.
            If normalised, the return numpy array type is float32 instead of uint8.

    Returns:
        frames as a NxHxWxC array (Number of frames x Height x Width x Channels)
    """
    vidcap = cv2.VideoCapture(video_filename)
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    success, image = vidcap.read()

    frame_type = np.uint8
    if normalise:
        frame_type = np.float32
    frames = np.zeros([n_frames] + list(image.shape), dtype=frame_type)

    count = 0
    while success:
        if normalise:
            image = image.astype(frame_type) / 255
        frames[count, ...] = image
        success, image = vidcap.read()
        count += 1

    return frames
