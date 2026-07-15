"""
video_utils.py
--------------
Placeholder for video and frame utility functions.

Responsibilities (Day 2):
- Open video sources (webcam, file, RTSP stream)
- Resize and preprocess frames
- Annotate frames with bounding boxes and track IDs
"""


def open_video_source(source):
    """
    Open a video capture source.

    Args:
        source (int | str): Camera index or file/stream path.

    Returns:
        cv2.VideoCapture: Opened capture object.
    """
    pass


def resize_frame(frame, width: int, height: int):
    """
    Resize a frame to the target dimensions.

    Args:
        frame: numpy.ndarray — BGR frame.
        width (int): Target width.
        height (int): Target height.

    Returns:
        numpy.ndarray: Resized frame.
    """
    pass


def draw_tracks(frame, tracked_objects: list):
    """
    Draw bounding boxes and track IDs on a frame.

    Args:
        frame: numpy.ndarray — BGR frame.
        tracked_objects (list[dict]): Tracked object list.

    Returns:
        numpy.ndarray: Annotated frame.
    """
    pass
