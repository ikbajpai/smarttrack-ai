"""
test_tracker.py
---------------
Unit tests for Tracker and TrackedObject.

BYTETracker is mocked in all tests so that no YOLO weights or GPU are needed.
Integration against a real video source is covered by the standalone
test_tracker.py script in the project root.
"""

from __future__ import annotations

from argparse import Namespace
from collections import deque
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.detection.detector import Detection
from src.tracking.tracker import Tracker, TrackedObject, _BoxProxy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRACKER_CFG: dict = {
    "type": "bytetrack",
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.6,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "frame_rate": 30,
    "trail_max_length": 10,
    "trail_thickness": 2,
    "trail_colour": [0, 255, 255],
}

BLANK_FRAME: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)


def _make_detection(x1=100, y1=100, x2=200, y2=300, conf=0.8) -> Detection:
    return Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=0, class_name="person")


def _make_raw_output_row(track_id: int, x1=100, y1=100, x2=200, y2=300, score=0.8) -> np.ndarray:
    """Return one row of BYTETracker._format_output(): [x1,y1,x2,y2,track_id,score,cls,idx]."""
    return np.array([x1, y1, x2, y2, track_id, score, 0, 0], dtype=np.float32)


@pytest.fixture()
def tracker() -> Tracker:
    """Return a Tracker whose underlying BYTETracker is mocked."""
    with patch("src.tracking.tracker.BYTETracker") as mock_bt_cls:
        mock_bt_cls.return_value = MagicMock()
        t = Tracker(TRACKER_CFG)
    return t


# ---------------------------------------------------------------------------
# TrackedObject
# ---------------------------------------------------------------------------

class TestTrackedObject:
    def test_fields(self):
        obj = TrackedObject(
            track_id=1,
            bbox=(10, 20, 100, 200),
            confidence=0.9,
            class_name="person",
            centroid=(55, 110),
        )
        assert obj.track_id == 1
        assert obj.bbox == (10, 20, 100, 200)
        assert obj.confidence == 0.9
        assert obj.class_name == "person"
        assert obj.centroid == (55, 110)

    def test_immutable(self):
        obj = TrackedObject(1, (0, 0, 10, 10), 0.5, "person", (5, 5))
        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            obj.track_id = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _BoxProxy
# ---------------------------------------------------------------------------

class TestBoxProxy:
    def test_attributes_present(self):
        xywh = np.zeros((3, 4), dtype=np.float32)
        conf = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        cls = np.zeros(3, dtype=np.float32)
        proxy = _BoxProxy(xywh, conf, cls)
        assert proxy.xywh is xywh
        assert proxy.conf is conf
        assert proxy.cls is cls

    def test_empty_arrays(self):
        proxy = _BoxProxy(
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
        assert proxy.xywh.shape == (0, 4)
        assert proxy.conf.shape == (0,)

    def test_boolean_indexing(self):
        xywh = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        conf = np.array([0.9, 0.5], dtype=np.float32)
        cls = np.zeros(2, dtype=np.float32)
        proxy = _BoxProxy(xywh, conf, cls)
        mask = np.array([True, False])
        subset = proxy[mask]
        assert len(subset) == 1
        np.testing.assert_array_equal(subset.xywh, xywh[[True, False]])

    def test_len(self):
        proxy = _BoxProxy(np.zeros((5, 4)), np.zeros(5), np.zeros(5))
        assert len(proxy) == 5


# ---------------------------------------------------------------------------
# Tracker.__init__
# ---------------------------------------------------------------------------

class TestTrackerInit:
    def test_loads_config_values(self, tracker: Tracker):
        assert tracker._track_high_thresh == 0.5
        assert tracker._track_low_thresh == 0.1
        assert tracker._new_track_thresh == 0.6
        assert tracker._track_buffer == 30
        assert tracker._match_thresh == 0.8
        assert tracker._frame_rate == 30
        assert tracker._trail_length == 10
        assert tracker._trail_thickness == 2
        assert tracker._trail_colour == (0, 255, 255)

    def test_trails_empty_at_start(self, tracker: Tracker):
        assert tracker._trails == {}
        assert tracker.active_trail_count == 0

    def test_defaults_applied_for_missing_keys(self):
        with patch("src.tracking.tracker.BYTETracker"):
            t = Tracker({})  # empty config — all defaults
        assert t._track_high_thresh == 0.5
        assert t._trail_length == 30
        assert t._trail_colour == (0, 255, 255)


# ---------------------------------------------------------------------------
# Tracker.update
# ---------------------------------------------------------------------------

class TestTrackerUpdate:
    def test_returns_empty_on_none_frame(self, tracker: Tracker):
        result = tracker.update([_make_detection()], None)  # type: ignore[arg-type]
        assert result == []

    def test_returns_empty_on_empty_frame(self, tracker: Tracker):
        result = tracker.update([_make_detection()], np.empty(0, dtype=np.uint8))
        assert result == []

    def test_returns_tracked_objects(self, tracker: Tracker):
        row = _make_raw_output_row(track_id=7, x1=100, y1=100, x2=200, y2=300, score=0.85)
        tracker._byte_tracker.update.return_value = np.array([row])

        result = tracker.update([_make_detection()], BLANK_FRAME)

        assert len(result) == 1
        obj = result[0]
        assert isinstance(obj, TrackedObject)
        assert obj.track_id == 7
        assert obj.bbox == (100, 100, 200, 300)
        assert obj.confidence == pytest.approx(0.85)
        assert obj.class_name == "person"
        assert obj.centroid == (150, 200)

    def test_centroid_calculation(self, tracker: Tracker):
        row = _make_raw_output_row(track_id=1, x1=0, y1=0, x2=100, y2=60)
        tracker._byte_tracker.update.return_value = np.array([row])

        result = tracker.update([], BLANK_FRAME)
        assert result[0].centroid == (50, 30)

    def test_trails_populated_after_update(self, tracker: Tracker):
        row = _make_raw_output_row(track_id=3)
        tracker._byte_tracker.update.return_value = np.array([row])

        tracker.update([], BLANK_FRAME)

        assert 3 in tracker._trails
        assert len(tracker._trails[3]) == 1

    def test_trails_accumulate_across_frames(self, tracker: Tracker):
        row = _make_raw_output_row(track_id=5)
        tracker._byte_tracker.update.return_value = np.array([row])

        for _ in range(5):
            tracker.update([], BLANK_FRAME)

        assert len(tracker._trails[5]) == 5

    def test_trail_bounded_by_max_length(self, tracker: Tracker):
        row = _make_raw_output_row(track_id=2)
        tracker._byte_tracker.update.return_value = np.array([row])

        for _ in range(tracker._trail_length + 20):
            tracker.update([], BLANK_FRAME)

        assert len(tracker._trails[2]) == tracker._trail_length

    def test_returns_empty_on_bytetrack_exception(self, tracker: Tracker):
        tracker._byte_tracker.update.side_effect = RuntimeError("tracker crash")
        result = tracker.update([_make_detection()], BLANK_FRAME)
        assert result == []

    def test_empty_detections_passed_to_bytetrack(self, tracker: Tracker):
        tracker._byte_tracker.update.return_value = np.empty((0, 8), dtype=np.float32)
        result = tracker.update([], BLANK_FRAME)
        assert result == []
        # Verify the proxy passed had empty xywh arrays
        call_args = tracker._byte_tracker.update.call_args
        proxy = call_args[0][0]
        assert proxy.xywh.shape[0] == 0

    def test_multiple_tracks_returned(self, tracker: Tracker):
        rows = np.array([
            _make_raw_output_row(track_id=1, x1=0, y1=0, x2=50, y2=100),
            _make_raw_output_row(track_id=2, x1=200, y1=200, x2=300, y2=400),
        ])
        tracker._byte_tracker.update.return_value = rows

        result = tracker.update([], BLANK_FRAME)

        assert len(result) == 2
        ids = {obj.track_id for obj in result}
        assert ids == {1, 2}


# ---------------------------------------------------------------------------
# Tracker.draw_annotations
# ---------------------------------------------------------------------------

class TestTrackerDrawAnnotations:
    def _obj(self, track_id: int = 1) -> TrackedObject:
        return TrackedObject(
            track_id=track_id,
            bbox=(50, 50, 150, 200),
            confidence=0.9,
            class_name="person",
            centroid=(100, 125),
        )

    def test_returns_copy_not_original(self, tracker: Tracker):
        frame = BLANK_FRAME.copy()
        result = tracker.draw_annotations(frame, [])
        assert result is not frame

    def test_original_frame_unmodified(self, tracker: Tracker):
        frame = BLANK_FRAME.copy()
        original = frame.copy()
        tracker.draw_annotations(frame, [self._obj()])
        np.testing.assert_array_equal(frame, original)

    def test_returns_ndarray(self, tracker: Tracker):
        result = tracker.draw_annotations(BLANK_FRAME, [self._obj()])
        assert isinstance(result, np.ndarray)

    def test_annotation_modifies_output(self, tracker: Tracker):
        result = tracker.draw_annotations(BLANK_FRAME, [self._obj()])
        # A completely blank frame plus a box should produce non-zero pixels.
        assert result.sum() > 0

    def test_show_count_false_skips_hud(self, tracker: Tracker):
        result_with = tracker.draw_annotations(BLANK_FRAME, [], show_count=True)
        result_without = tracker.draw_annotations(BLANK_FRAME, [], show_count=False)
        # With HUD text there will be more non-zero pixels.
        assert result_with.sum() >= result_without.sum()


# ---------------------------------------------------------------------------
# Tracker.reset
# ---------------------------------------------------------------------------

class TestTrackerReset:
    def test_clears_trails(self, tracker: Tracker):
        tracker._trails[1] = deque([(10, 10), (20, 20)])
        tracker.reset()
        assert tracker._trails == {}

    def test_reinitialises_bytetracker(self, tracker: Tracker):
        old_bt = tracker._byte_tracker
        tracker.reset()
        # A fresh BYTETracker instance should have been created.
        assert tracker._byte_tracker is not old_bt


# ---------------------------------------------------------------------------
# Tracker._track_colour
# ---------------------------------------------------------------------------

class TestTrackColour:
    def test_returns_three_ints(self):
        colour = Tracker._track_colour(1)
        assert len(colour) == 3
        assert all(isinstance(c, int) for c in colour)

    def test_stable_across_calls(self):
        assert Tracker._track_colour(42) == Tracker._track_colour(42)

    def test_different_ids_different_colours(self):
        colours = {Tracker._track_colour(i) for i in range(1, 10)}
        assert len(colours) > 3  # most IDs should produce distinct colours

    def test_values_in_bgr_range(self):
        for track_id in range(0, 50):
            colour = Tracker._track_colour(track_id)
            assert all(0 <= c <= 255 for c in colour)


# ---------------------------------------------------------------------------
# Tracker._build_proxy
# ---------------------------------------------------------------------------

class TestBuildProxy:
    def test_empty_detections(self, tracker: Tracker):
        proxy = tracker._build_proxy([])
        assert proxy.xywh.shape == (0, 4)
        assert proxy.conf.shape == (0,)
        assert proxy.cls.shape == (0,)

    def test_single_detection_center_format(self, tracker: Tracker):
        # xyxy: x1=10, y1=20, x2=110, y2=220  →  cx=60, cy=120, w=100, h=200
        det = _make_detection(x1=10, y1=20, x2=110, y2=220, conf=0.75)
        proxy = tracker._build_proxy([det])
        assert proxy.xywh.shape == (1, 4)
        np.testing.assert_allclose(proxy.xywh[0], [60.0, 120.0, 100.0, 200.0])
        assert float(proxy.conf[0]) == pytest.approx(0.75)
        assert float(proxy.cls[0]) == 0.0

    def test_multiple_detections(self, tracker: Tracker):
        dets = [_make_detection(conf=0.9), _make_detection(x1=300, conf=0.6)]
        proxy = tracker._build_proxy(dets)
        assert proxy.xywh.shape == (2, 4)
        assert proxy.conf.shape == (2,)

    def test_proxy_supports_boolean_indexing(self, tracker: Tracker):
        dets = [_make_detection(conf=0.9), _make_detection(x1=300, conf=0.3)]
        proxy = tracker._build_proxy(dets)
        mask = proxy.conf >= 0.5
        subset = proxy[mask]
        assert len(subset) == 1
