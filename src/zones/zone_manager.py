"""
zone_manager.py
---------------
Placeholder for restricted zone definition and intrusion logic.

Responsibilities (Day 3):
- Load zone polygon definitions from config or UI input
- Check whether a tracked object's bounding box centroid falls inside any zone
- Return intrusion events with track ID, zone ID, and timestamp
"""


class ZoneManager:
    """Manages restricted zone polygons and intrusion detection logic."""

    def __init__(self, zones_config: list):
        """
        Initialize with list of zone definitions from config.yaml.

        Args:
            zones_config (list): Zone entries from config.yaml.
        """
        pass

    def load_zones(self) -> None:
        """Parse and store zone polygons."""
        pass

    def check_intrusion(self, tracked_objects: list) -> list:
        """
        Determine which tracked objects are inside restricted zones.

        Args:
            tracked_objects (list[dict]): Output from Tracker.update().

        Returns:
            list[dict]: Intrusion events with keys:
                        'track_id', 'zone_id', 'zone_name', 'timestamp'.
        """
        pass

    def draw_zones(self, frame):
        """
        Overlay zone polygons on the given frame.

        Args:
            frame: numpy.ndarray — BGR frame.

        Returns:
            numpy.ndarray: Frame with zones drawn.
        """
        pass
