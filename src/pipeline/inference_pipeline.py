"""
inference_pipeline.py
---------------------
Placeholder for the central inference orchestration pipeline.

Responsibilities (Day 4):
- Accept a video frame as input
- Run detect → track → zone_check → alert in sequence
- Return an annotated frame and any triggered events
"""


class InferencePipeline:
    """
    Orchestrates the full detection → tracking → zone intrusion pipeline.
    """

    def __init__(self, config: dict):
        """
        Initialize all sub-components from unified config.

        Args:
            config (dict): Full parsed config.yaml.
        """
        pass

    def setup(self) -> None:
        """Instantiate and configure Detector, Tracker, ZoneManager, AlertManager."""
        pass

    def process_frame(self, frame):
        """
        Run the full pipeline on a single frame.

        Args:
            frame: numpy.ndarray — BGR frame from OpenCV.

        Returns:
            tuple: (annotated_frame, intrusion_events)
        """
        pass

    def teardown(self) -> None:
        """Release resources cleanly."""
        pass
