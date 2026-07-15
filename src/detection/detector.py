"""
detector.py
-----------
Placeholder for YOLO-based object detection.

Responsibilities (Day 2):
- Load Ultralytics YOLO model from config
- Run inference on a single frame
- Return bounding boxes, class labels, and confidence scores
"""


class Detector:
    """Wraps an Ultralytics YOLO model for object detection."""

    def __init__(self, config: dict):
        """
        Initialize the detector with model config.

        Args:
            config (dict): Model section from config.yaml.
        """
        pass

    def load(self) -> None:
        """Load model weights from disk."""
        pass

    def detect(self, frame):
        """
        Run detection on a single BGR frame.

        Args:
            frame: numpy.ndarray — BGR image from OpenCV.

        Returns:
            list[dict]: List of detections with keys:
                        'bbox' (x1,y1,x2,y2), 'confidence', 'class_name'.
        """
        pass
