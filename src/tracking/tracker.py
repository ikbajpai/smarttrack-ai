"""
tracker.py
----------
Placeholder for ByteTrack multi-object tracking.

Responsibilities (Day 2):
- Accept detection results per frame
- Assign and maintain persistent track IDs across frames
- Return tracked objects with stable identities
"""


class Tracker:
    """Wraps ByteTrack for multi-object tracking."""

    def __init__(self, config: dict):
        """
        Initialize the tracker with tracker config.

        Args:
            config (dict): Tracker section from config.yaml.
        """
        pass

    def update(self, detections, frame):
        """
        Update tracker state with new detections.

        Args:
            detections (list[dict]): Output from Detector.detect().
            frame: numpy.ndarray — current BGR frame.

        Returns:
            list[dict]: Tracked objects with keys:
                        'track_id', 'bbox', 'class_name'.
        """
        pass

    def reset(self) -> None:
        """Reset tracker state between video sources."""
        pass
