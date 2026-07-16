"""
test_zone_manager.py
--------------------
Unit tests for ZoneManager, IntrusionEvent, and EventType.

All tests are self-contained and require no video files, YOLO weights, or
running services.  Shapely is the only non-stdlib dependency exercised here.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.tracking.tracker import TrackedObject
from src.zones.zone_manager import (
    EventType,
    IntrusionEvent,
    ZoneManager,
    _build_polygon,
    _parse_color,
    _resolve_probe,
)

# ---------------------------------------------------------------------------
# Shared geometry fixtures
# ---------------------------------------------------------------------------

# A 300×300 square zone anchored at (100, 100)
SQUARE_POINTS: list[tuple[int, int]] = [
    (100, 100), (400, 100), (400, 400), (100, 400)
]

# A zone that does not overlap SQUARE_POINTS
FAR_POINTS: list[tuple[int, int]] = [
    (600, 600), (900, 600), (900, 900), (600, 900)
]

EMPTY_CONFIG: list = []


def _make_obj(
    track_id: int,
    x: int,
    y: int,
    *,
    w: int = 50,
    h: int = 100,
) -> TrackedObject:
    """Create a TrackedObject whose foot_point is at (x, y).

    The bounding box is built so that bottom-centre == (x, y):
        x1 = x - w//2, y1 = y - h, x2 = x + w//2, y2 = y
    """
    x1, y1, x2, y2 = x - w // 2, y - h, x + w // 2, y
    return TrackedObject(
        track_id=track_id,
        bbox=(x1, y1, x2, y2),
        confidence=0.9,
        class_name="person",
        centroid=(x, y - h // 2),
        foot_point=(x, y),
    )


@pytest.fixture()
def zm() -> ZoneManager:
    """ZoneManager with a single square zone and no config-file zones."""
    manager = ZoneManager(EMPTY_CONFIG)
    manager.add_zone("Zone A", SQUARE_POINTS)
    return manager


# ---------------------------------------------------------------------------
# IntrusionEvent
# ---------------------------------------------------------------------------

class TestIntrusionEvent:
    def test_fields(self):
        ts = datetime.utcnow()
        ev = IntrusionEvent(
            timestamp=ts,
            track_id=7,
            zone_name="Zone A",
            event_type=EventType.ENTER,
        )
        assert ev.track_id == 7
        assert ev.zone_name == "Zone A"
        assert ev.event_type == EventType.ENTER
        assert ev.timestamp is ts

    def test_immutable(self):
        ev = IntrusionEvent(datetime.utcnow(), 1, "Z", EventType.ENTER)
        with pytest.raises(Exception):
            ev.track_id = 99  # type: ignore[misc]

    def test_event_type_values(self):
        assert EventType.ENTER == "ENTER"
        assert EventType.EXIT == "EXIT"


# ---------------------------------------------------------------------------
# TrackedObject.foot_point (backward compatibility + new field)
# ---------------------------------------------------------------------------

class TestTrackedObjectFootPoint:
    def test_foot_point_populated_by_tracker_update(self):
        """foot_point should be (cx, y2) — verified via helper."""
        obj = _make_obj(1, x=250, y=400)
        assert obj.foot_point == (250, 400)

    def test_foot_point_defaults_to_none_for_legacy(self):
        """Existing five-arg construction must not break."""
        obj = TrackedObject(1, (0, 0, 100, 200), 0.8, "person", (50, 100))
        assert obj.foot_point is None

    def test_resolve_probe_prefers_foot_point(self):
        obj = _make_obj(1, 250, 400)
        assert _resolve_probe(obj) == (250, 400)

    def test_resolve_probe_falls_back_to_centroid(self):
        obj = TrackedObject(1, (0, 0, 100, 200), 0.8, "person", (50, 100))
        assert _resolve_probe(obj) == (50, 100)


# ---------------------------------------------------------------------------
# ZoneManager.__init__ and add_zone / remove_zone
# ---------------------------------------------------------------------------

class TestZoneManagerInit:
    def test_empty_config_no_zones(self):
        manager = ZoneManager(EMPTY_CONFIG)
        assert manager.zone_count == 0
        assert manager.zone_names == []

    def test_config_with_polygon_loaded(self):
        cfg = [{"name": "Z1", "polygon": SQUARE_POINTS, "color": [0, 0, 255]}]
        manager = ZoneManager(cfg)
        assert manager.zone_count == 1
        assert "Z1" in manager.zone_names

    def test_config_with_empty_polygon_skipped(self):
        cfg = [{"name": "Empty", "polygon": []}]
        manager = ZoneManager(cfg)
        assert manager.zone_count == 0

    def test_config_with_missing_polygon_key_skipped(self):
        cfg = [{"name": "No poly"}]
        manager = ZoneManager(cfg)
        assert manager.zone_count == 0


class TestAddZone:
    def test_add_valid_zone(self):
        manager = ZoneManager(EMPTY_CONFIG)
        result = manager.add_zone("Z", SQUARE_POINTS)
        assert result is True
        assert manager.zone_count == 1

    def test_add_zone_with_too_few_points_rejected(self):
        manager = ZoneManager(EMPTY_CONFIG)
        result = manager.add_zone("Z", [(0, 0), (100, 0)])
        assert result is False
        assert manager.zone_count == 0

    def test_add_zone_replaces_duplicate_name(self):
        manager = ZoneManager(EMPTY_CONFIG)
        manager.add_zone("Z", SQUARE_POINTS)
        manager.add_zone("Z", FAR_POINTS)  # replace
        assert manager.zone_count == 1

    def test_add_multiple_zones(self):
        manager = ZoneManager(EMPTY_CONFIG)
        manager.add_zone("A", SQUARE_POINTS)
        manager.add_zone("B", FAR_POINTS)
        assert manager.zone_count == 2
        assert set(manager.zone_names) == {"A", "B"}


class TestRemoveZone:
    def test_remove_existing_zone(self, zm: ZoneManager):
        result = zm.remove_zone("Zone A")
        assert result is True
        assert zm.zone_count == 0

    def test_remove_nonexistent_zone(self, zm: ZoneManager):
        result = zm.remove_zone("Does Not Exist")
        assert result is False

    def test_remove_clears_active_intrusions(self, zm: ZoneManager):
        # Artificially inject an active intrusion for the zone.
        zm._active_intrusions.add((42, "Zone A"))
        zm.remove_zone("Zone A")
        assert (42, "Zone A") not in zm._active_intrusions


# ---------------------------------------------------------------------------
# ZoneManager.check_intrusions — ENTER / EXIT logic
# ---------------------------------------------------------------------------

class TestCheckIntrusionsEnter:
    def test_person_inside_zone_emits_enter(self, zm: ZoneManager):
        obj = _make_obj(1, x=250, y=300)   # inside SQUARE_POINTS
        events = zm.check_intrusions([obj])
        assert len(events) == 1
        assert events[0].event_type == EventType.ENTER
        assert events[0].track_id == 1
        assert events[0].zone_name == "Zone A"

    def test_enter_not_repeated_while_inside(self, zm: ZoneManager):
        obj = _make_obj(1, x=250, y=300)
        zm.check_intrusions([obj])
        events = zm.check_intrusions([obj])  # second call — already inside
        assert events == []

    def test_person_outside_zone_no_event(self, zm: ZoneManager):
        obj = _make_obj(1, x=50, y=50)   # outside SQUARE_POINTS
        events = zm.check_intrusions([obj])
        assert events == []

    def test_event_has_utc_timestamp(self, zm: ZoneManager):
        obj = _make_obj(1, x=250, y=300)
        before = datetime.utcnow()
        events = zm.check_intrusions([obj])
        after = datetime.utcnow()
        assert before <= events[0].timestamp <= after


class TestCheckIntrusionsExit:
    def test_person_leaving_zone_emits_exit(self, zm: ZoneManager):
        inside = _make_obj(1, x=250, y=300)
        outside = _make_obj(1, x=50, y=50)

        zm.check_intrusions([inside])           # ENTER
        events = zm.check_intrusions([outside]) # EXIT

        assert len(events) == 1
        assert events[0].event_type == EventType.EXIT
        assert events[0].track_id == 1

    def test_track_dropped_emits_exit(self, zm: ZoneManager):
        inside = _make_obj(1, x=250, y=300)
        zm.check_intrusions([inside])           # ENTER

        events = zm.check_intrusions([])        # track disappeared
        assert len(events) == 1
        assert events[0].event_type == EventType.EXIT
        assert events[0].track_id == 1

    def test_exit_clears_active_intrusion(self, zm: ZoneManager):
        inside = _make_obj(1, x=250, y=300)
        outside = _make_obj(1, x=50, y=50)

        zm.check_intrusions([inside])
        zm.check_intrusions([outside])

        assert zm.active_intrusion_count == 0

    def test_re_enter_after_exit_emits_enter_again(self, zm: ZoneManager):
        inside = _make_obj(1, x=250, y=300)
        outside = _make_obj(1, x=50, y=50)

        zm.check_intrusions([inside])   # ENTER
        zm.check_intrusions([outside])  # EXIT
        events = zm.check_intrusions([inside])  # ENTER again

        assert len(events) == 1
        assert events[0].event_type == EventType.ENTER


class TestCheckIntrusionsMultiple:
    def test_multiple_persons_multiple_events(self, zm: ZoneManager):
        a = _make_obj(1, x=250, y=300)  # inside
        b = _make_obj(2, x=250, y=300)  # inside
        events = zm.check_intrusions([a, b])
        assert len(events) == 2
        assert all(ev.event_type == EventType.ENTER for ev in events)

    def test_multiple_zones_each_person_tested_against_all(self):
        manager = ZoneManager(EMPTY_CONFIG)
        manager.add_zone("A", SQUARE_POINTS)
        manager.add_zone("B", FAR_POINTS)

        obj = _make_obj(1, x=250, y=300)   # inside A, outside B
        events = manager.check_intrusions([obj])

        assert len(events) == 1
        assert events[0].zone_name == "A"

    def test_no_zones_returns_empty(self):
        manager = ZoneManager(EMPTY_CONFIG)
        obj = _make_obj(1, x=250, y=300)
        assert manager.check_intrusions([obj]) == []

    def test_empty_tracked_objects_no_events(self, zm: ZoneManager):
        assert zm.check_intrusions([]) == []

    def test_active_intrusion_count_tracks_state(self, zm: ZoneManager):
        a = _make_obj(1, x=250, y=300)
        b = _make_obj(2, x=250, y=300)
        zm.check_intrusions([a, b])
        assert zm.active_intrusion_count == 2

        zm.check_intrusions([])  # both tracks lost
        assert zm.active_intrusion_count == 0


# ---------------------------------------------------------------------------
# ZoneManager.reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_intrusion_state(self, zm: ZoneManager):
        obj = _make_obj(1, x=250, y=300)
        zm.check_intrusions([obj])
        zm.reset()
        assert zm.active_intrusion_count == 0

    def test_reset_preserves_zones(self, zm: ZoneManager):
        zm.reset()
        assert zm.zone_count == 1

    def test_enter_emitted_again_after_reset(self, zm: ZoneManager):
        obj = _make_obj(1, x=250, y=300)
        zm.check_intrusions([obj])  # ENTER recorded
        zm.reset()
        events = zm.check_intrusions([obj])  # should fire again
        assert len(events) == 1
        assert events[0].event_type == EventType.ENTER


# ---------------------------------------------------------------------------
# ZoneManager.load_zones_from_file / save_zones_to_file
# ---------------------------------------------------------------------------

class TestFileIO:
    def _write_json(self, tmp_path: Path, data: dict) -> Path:
        p = tmp_path / "zones.json"
        p.write_text(json.dumps(data))
        return p

    def test_load_valid_file(self, tmp_path: Path):
        p = self._write_json(tmp_path, {
            "zones": [{"name": "Z", "polygon": SQUARE_POINTS, "color": [0, 0, 255]}]
        })
        manager = ZoneManager(EMPTY_CONFIG)
        count = manager.load_zones_from_file(p)
        assert count == 1
        assert "Z" in manager.zone_names

    def test_load_nonexistent_file_returns_zero(self):
        manager = ZoneManager(EMPTY_CONFIG)
        result = manager.load_zones_from_file("/nonexistent/path/zones.json")
        assert result == 0

    def test_load_malformed_json_returns_zero(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{ not valid json }")
        manager = ZoneManager(EMPTY_CONFIG)
        result = manager.load_zones_from_file(p)
        assert result == 0

    def test_load_missing_polygon_skipped(self, tmp_path: Path):
        p = self._write_json(tmp_path, {
            "zones": [{"name": "Z", "polygon": []}]
        })
        manager = ZoneManager(EMPTY_CONFIG)
        count = manager.load_zones_from_file(p)
        assert count == 0

    def test_save_and_reload_roundtrip(self, tmp_path: Path):
        manager = ZoneManager(EMPTY_CONFIG)
        manager.add_zone("RoundTrip", SQUARE_POINTS, color=(10, 20, 30), alert_cooldown_seconds=7)
        save_path = tmp_path / "rt.json"
        manager.save_zones_to_file(save_path)

        reloaded = ZoneManager(EMPTY_CONFIG)
        reloaded.load_zones_from_file(save_path)

        assert reloaded.zone_count == 1
        assert "RoundTrip" in reloaded.zone_names

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        manager = ZoneManager(EMPTY_CONFIG)
        manager.add_zone("Z", SQUARE_POINTS)
        deep_path = tmp_path / "a" / "b" / "c" / "zones.json"
        manager.save_zones_to_file(deep_path)
        assert deep_path.exists()

    def test_load_invalid_zones_key_type(self, tmp_path: Path):
        p = self._write_json(tmp_path, {"zones": "not a list"})
        manager = ZoneManager(EMPTY_CONFIG)
        result = manager.load_zones_from_file(p)
        assert result == 0


# ---------------------------------------------------------------------------
# ZoneManager.draw_zones
# ---------------------------------------------------------------------------

class TestDrawZones:
    def test_returns_copy_not_original(self, zm: ZoneManager):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = zm.draw_zones(frame)
        assert result is not frame

    def test_original_frame_unmodified(self, zm: ZoneManager):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        original = frame.copy()
        zm.draw_zones(frame)
        np.testing.assert_array_equal(frame, original)

    def test_overlay_produces_non_zero_pixels(self, zm: ZoneManager):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = zm.draw_zones(frame)
        assert result.sum() > 0

    def test_no_zones_returns_identical_copy(self):
        manager = ZoneManager(EMPTY_CONFIG)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = manager.draw_zones(frame)
        np.testing.assert_array_equal(result, frame)
        assert result is not frame


# ---------------------------------------------------------------------------
# _build_polygon helper
# ---------------------------------------------------------------------------

class TestBuildPolygon:
    def test_valid_polygon(self):
        poly = _build_polygon("Z", SQUARE_POINTS)
        assert poly is not None
        assert poly.area > 0

    def test_too_few_points_returns_none(self):
        assert _build_polygon("Z", [(0, 0), (1, 1)]) is None

    def test_zero_area_polygon_returns_none(self):
        # Collinear points — zero area.
        assert _build_polygon("Z", [(0, 0), (1, 0), (2, 0)]) is None

    def test_triangle_is_valid(self):
        poly = _build_polygon("Z", [(0, 0), (100, 0), (50, 100)])
        assert poly is not None


# ---------------------------------------------------------------------------
# _parse_color helper
# ---------------------------------------------------------------------------

class TestParseColor:
    def test_valid_list(self):
        assert _parse_color([0, 128, 255]) == (0, 128, 255)

    def test_none_returns_default(self):
        assert _parse_color(None) == (0, 0, 255)

    def test_clamped_to_255(self):
        r, g, b = _parse_color([300, -10, 128])
        assert r == 255
        assert g == 0
        assert b == 128

    def test_invalid_type_returns_default(self):
        assert _parse_color("red") == (0, 0, 255)
