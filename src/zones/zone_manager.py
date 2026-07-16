"""
zone_manager.py
---------------
Restricted zone polygon management and intrusion detection for SmartTrack AI.

Provides:
    EventType      — Enum distinguishing ENTER from EXIT intrusion events.
    IntrusionEvent — Frozen dataclass emitted when a tracked person crosses a
                     zone boundary.
    ZoneManager    — Manages named polygon zones, detects intrusions using
                     Shapely point-in-polygon tests, and returns structured
                     events for downstream alert handling.

This module depends only on TrackedObject from tracker.py.  It never imports
Detector, AlertManager, or Streamlit.  The pipeline layer wires modules together.

Usage::

    import yaml
    from src.zones.zone_manager import ZoneManager

    with open("config/config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    zm = ZoneManager(cfg["zones"])

    # Per-frame call inside the video loop:
    events = zm.check_intrusions(tracked_objects)
    annotated = zm.draw_zones(frame)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid

from src.tracking.tracker import TrackedObject

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_POLYGON_VERTICES: int = 3
_DEFAULT_ZONE_COLOR: tuple[int, int, int] = (0, 0, 255)    # red BGR
_DEFAULT_COOLDOWN_SECONDS: float = 5.0
_ZONE_ALPHA: float = 0.25                                   # fill opacity


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Discriminates between zone-entry and zone-exit intrusion events.

    Using ``str`` as a mixin makes the enum JSON-serialisable without a
    custom encoder.
    """

    ENTER = "ENTER"
    EXIT = "EXIT"


@dataclass(frozen=True)
class IntrusionEvent:
    """An immutable record emitted when a tracked person crosses a zone boundary.

    Attributes:
        timestamp: UTC wall-clock time when the event was generated.
        track_id: The ByteTrack integer ID of the person who triggered the event.
        zone_name: Human-readable name of the zone that was entered or exited.
        event_type: :class:`EventType.ENTER` or :class:`EventType.EXIT`.
    """

    timestamp: datetime
    track_id: int
    zone_name: str
    event_type: EventType


# ---------------------------------------------------------------------------
# Internal zone record
# ---------------------------------------------------------------------------

@dataclass
class _ZoneRecord:
    """Internal representation of a single restricted zone.

    Attributes:
        name: Unique human-readable identifier.
        polygon: Shapely :class:`~shapely.geometry.Polygon` used for
            point-in-polygon tests.
        points: Original ``[[x, y], ...]`` list retained for serialisation and
            OpenCV drawing (avoids round-tripping through Shapely).
        color: BGR tuple for overlay rendering.
        alert_cooldown_seconds: Minimum seconds between successive ENTER events
            for the same ``(track_id, zone_name)`` pair (reserved for
            AlertManager; ZoneManager itself enforces one ENTER until EXIT).
    """

    name: str
    polygon: Polygon
    points: list[tuple[int, int]]
    color: tuple[int, int, int]
    alert_cooldown_seconds: float


# ---------------------------------------------------------------------------
# ZoneManager
# ---------------------------------------------------------------------------

class ZoneManager:
    """Manages restricted polygon zones and detects person intrusions.

    Zones can be loaded from the ``zones`` section of ``config/config.yaml``
    and/or from JSON files in ``data/zone_configs/``.  Intrusion state is
    maintained per ``(track_id, zone_name)`` pair so that a single ENTER event
    fires on first contact and a single EXIT fires when the person leaves or
    their track is dropped by ByteTrack.

    Args:
        zones_config: The ``zones`` list from ``config/config.yaml``.  Each
            entry may define a polygon or leave it empty (to be filled later
            via the UI or :meth:`load_zones_from_file`).  Pass an empty list
            ``[]`` to start with no zones.

    Example::

        import yaml
        from src.zones.zone_manager import ZoneManager

        with open("config/config.yaml") as fh:
            cfg = yaml.safe_load(fh)

        zm = ZoneManager(cfg["zones"])
        zm.load_zones_from_file("data/zone_configs/warehouse.json")

        events = zm.check_intrusions(tracked_objects)
    """

    def __init__(self, zones_config: list[dict[str, Any]]) -> None:
        self._zones: list[_ZoneRecord] = []
        # Set of (track_id, zone_name) pairs currently inside a zone.
        self._active_intrusions: set[tuple[int, str]] = set()

        self._load_zones_from_config(zones_config)
        logger.info(
            "ZoneManager initialised with {} zone(s) from config.",
            len(self._zones),
        )

    # ------------------------------------------------------------------
    # Public API — zone management
    # ------------------------------------------------------------------

    def add_zone(
        self,
        name: str,
        points: list[tuple[int, int]],
        *,
        color: tuple[int, int, int] = _DEFAULT_ZONE_COLOR,
        alert_cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS,
    ) -> bool:
        """Add a new named polygon zone.

        If a zone with the same *name* already exists it is silently replaced.

        Args:
            name: Human-readable zone identifier; must be unique.
            points: Polygon vertices as ``[(x, y), ...]``.  At least three
                non-collinear points are required.
            color: BGR colour used when drawing the zone overlay.
            alert_cooldown_seconds: Seconds between successive ENTER events for
                the same person (reserved for AlertManager).

        Returns:
            ``True`` if the zone was added successfully, ``False`` if the
            polygon is invalid (too few points or degenerate geometry).
        """
        polygon = _build_polygon(name, points)
        if polygon is None:
            return False

        # Replace existing zone with the same name.
        self._zones = [z for z in self._zones if z.name != name]
        self._zones.append(
            _ZoneRecord(
                name=name,
                polygon=polygon,
                points=list(points),
                color=color,
                alert_cooldown_seconds=alert_cooldown_seconds,
            )
        )
        logger.info("Zone '{}' added ({} vertices).", name, len(points))
        return True

    def remove_zone(self, name: str) -> bool:
        """Remove the zone with the given *name*.

        Also clears any active intrusion state for that zone so that stale
        EXIT events are not emitted on the next :meth:`check_intrusions` call.

        Args:
            name: Zone identifier to remove.

        Returns:
            ``True`` if a zone was found and removed, ``False`` otherwise.
        """
        before = len(self._zones)
        self._zones = [z for z in self._zones if z.name != name]
        removed = len(self._zones) < before

        if removed:
            self._active_intrusions = {
                pair for pair in self._active_intrusions if pair[1] != name
            }
            logger.info("Zone '{}' removed.", name)
        else:
            logger.warning("remove_zone: zone '{}' not found.", name)

        return removed

    def load_zones_from_file(self, path: str | Path) -> int:
        """Load zone definitions from a JSON file and merge with existing zones.

        Zones in the file that share a name with an existing zone replace the
        existing entry.  Zones with invalid polygons are skipped with a warning.

        JSON schema::

            {
              "zones": [
                {
                  "name": "Restricted Area A",
                  "polygon": [[x1, y1], [x2, y2], ...],
                  "color": [0, 0, 255],
                  "alert_cooldown_seconds": 5
                }
              ]
            }

        Args:
            path: Path to the ``.json`` zone configuration file.

        Returns:
            Number of zones successfully loaded from the file.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("load_zones_from_file: '{}' not found.", path)
            return 0

        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.error("load_zones_from_file: malformed JSON in '{}': {}", path, exc)
            return 0

        raw_zones = data.get("zones", [])
        if not isinstance(raw_zones, list):
            logger.error(
                "load_zones_from_file: '{}' — 'zones' must be a list.", path
            )
            return 0

        loaded = 0
        for entry in raw_zones:
            if not isinstance(entry, dict):
                logger.warning("load_zones_from_file: skipping non-dict entry.")
                continue
            if self.add_zone(
                name=str(entry.get("name", f"zone_{loaded}")),
                points=[tuple(pt) for pt in entry.get("polygon", [])],  # type: ignore[misc]
                color=_parse_color(entry.get("color")),
                alert_cooldown_seconds=float(
                    entry.get("alert_cooldown_seconds", _DEFAULT_COOLDOWN_SECONDS)
                ),
            ):
                loaded += 1

        logger.info("Loaded {} zone(s) from '{}'.", loaded, path)
        return loaded

    def save_zones_to_file(self, path: str | Path) -> None:
        """Persist current zones to a JSON file.

        Creates parent directories as needed.  Overwrites an existing file.

        Args:
            path: Destination ``.json`` file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "zones": [
                {
                    "name": zone.name,
                    "polygon": [list(pt) for pt in zone.points],
                    "color": list(zone.color),
                    "alert_cooldown_seconds": zone.alert_cooldown_seconds,
                }
                for zone in self._zones
            ]
        }

        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        logger.info("Saved {} zone(s) to '{}'.", len(self._zones), path)

    # ------------------------------------------------------------------
    # Public API — intrusion detection
    # ------------------------------------------------------------------

    def check_intrusions(
        self, tracked_objects: list[TrackedObject]
    ) -> list[IntrusionEvent]:
        """Determine which tracked persons have entered or exited restricted zones.

        Called once per video frame with the output of
        :meth:`~src.tracking.tracker.Tracker.update`.

        Logic per frame:

        1. Any ``(track_id, zone_name)`` pair in ``_active_intrusions`` whose
           ``track_id`` is absent from *tracked_objects* (track dropped by
           ByteTrack) emits an EXIT event.
        2. For each tracked person and each zone:
           - If ``foot_point`` is inside the polygon and the pair is **not**
             already active → emit ENTER.
           - If ``foot_point`` is outside the polygon and the pair **is**
             active → emit EXIT.

        Args:
            tracked_objects: Output of :meth:`~src.tracking.tracker.Tracker.update`
                for the current frame.

        Returns:
            List of :class:`IntrusionEvent` instances (may be empty).  Events
            are ordered: lost-track EXITs first, then per-zone events in the
            order zones were registered.
        """
        if not self._zones:
            return []

        events: list[IntrusionEvent] = []
        now = datetime.utcnow()

        current_track_ids = {obj.track_id for obj in tracked_objects}

        # Step 1: emit EXIT for any track that ByteTrack dropped entirely.
        lost_pairs = {
            pair for pair in self._active_intrusions
            if pair[0] not in current_track_ids
        }
        for track_id, zone_name in lost_pairs:
            events.append(
                IntrusionEvent(
                    timestamp=now,
                    track_id=track_id,
                    zone_name=zone_name,
                    event_type=EventType.EXIT,
                )
            )
            logger.debug(
                "EXIT (track lost) | track_id={} zone='{}'", track_id, zone_name
            )
        self._active_intrusions -= lost_pairs

        # Step 2: per-person, per-zone point-in-polygon test.
        for obj in tracked_objects:
            probe = _resolve_probe(obj)
            shapely_point = Point(probe)

            for zone in self._zones:
                pair = (obj.track_id, zone.name)
                inside = zone.polygon.contains(shapely_point)

                if inside and pair not in self._active_intrusions:
                    self._active_intrusions.add(pair)
                    events.append(
                        IntrusionEvent(
                            timestamp=now,
                            track_id=obj.track_id,
                            zone_name=zone.name,
                            event_type=EventType.ENTER,
                        )
                    )
                    logger.info(
                        "ENTER | track_id={} zone='{}' probe={}",
                        obj.track_id, zone.name, probe,
                    )

                elif not inside and pair in self._active_intrusions:
                    self._active_intrusions.discard(pair)
                    events.append(
                        IntrusionEvent(
                            timestamp=now,
                            track_id=obj.track_id,
                            zone_name=zone.name,
                            event_type=EventType.EXIT,
                        )
                    )
                    logger.info(
                        "EXIT | track_id={} zone='{}' probe={}",
                        obj.track_id, zone.name, probe,
                    )

        return events

    def reset(self) -> None:
        """Clear all active intrusion state.

        Call this when switching video sources so that stale ``(track_id,
        zone_name)`` pairs from the previous clip do not trigger spurious EXIT
        events on the first frame of the new source.  Zone definitions are
        preserved.
        """
        self._active_intrusions.clear()
        logger.info("ZoneManager reset — intrusion state cleared.")

    # ------------------------------------------------------------------
    # Public API — rendering
    # ------------------------------------------------------------------

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Overlay zone polygons onto a copy of *frame*.

        Active zones (containing at least one person) are filled; inactive
        zones are drawn with a semi-transparent fill and a solid border.  The
        source *frame* is never mutated.

        Args:
            frame: BGR frame to annotate.

        Returns:
            A new BGR frame with all zone overlays applied.
        """
        annotated = frame.copy()

        for zone in self._zones:
            pts = np.array(zone.points, dtype=np.int32).reshape((-1, 1, 2))
            active_track_ids = {tid for tid, zn in self._active_intrusions if zn == zone.name}
            is_active = bool(active_track_ids)

            # Semi-transparent fill.
            overlay = annotated.copy()
            fill_color = zone.color if not is_active else (
                min(zone.color[0] + 80, 255),
                min(zone.color[1] + 80, 255),
                min(zone.color[2] + 80, 255),
            )
            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.addWeighted(overlay, _ZONE_ALPHA, annotated, 1 - _ZONE_ALPHA, 0, annotated)

            # Solid border.
            cv2.polylines(annotated, [pts], isClosed=True, color=zone.color, thickness=2)

            # Zone label.
            label_pt = _label_position(zone.points)
            cv2.putText(
                annotated,
                zone.name,
                label_pt,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                zone.color,
                2,
                cv2.LINE_AA,
            )

        return annotated

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def zone_names(self) -> list[str]:
        """Names of all currently registered zones, in registration order."""
        return [z.name for z in self._zones]

    @property
    def zone_count(self) -> int:
        """Number of currently registered zones."""
        return len(self._zones)

    @property
    def active_intrusion_count(self) -> int:
        """Number of ``(track_id, zone_name)`` pairs currently inside a zone."""
        return len(self._active_intrusions)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_zones_from_config(self, zones_config: list[dict[str, Any]]) -> None:
        """Parse the ``zones`` section of ``config/config.yaml``.

        Entries with empty or missing polygon lists are silently skipped (they
        are placeholders to be filled via the UI or JSON files).

        Args:
            zones_config: Raw list from ``cfg["zones"]``.
        """
        for entry in zones_config:
            if not isinstance(entry, dict):
                logger.warning("zones config: skipping non-dict entry: {}", entry)
                continue

            raw_polygon = entry.get("polygon", [])
            if not raw_polygon:
                logger.debug(
                    "zones config: zone '{}' has no polygon — skipped.",
                    entry.get("name", "<unnamed>"),
                )
                continue

            self.add_zone(
                name=str(entry.get("name", entry.get("id", "zone"))),
                points=[tuple(pt) for pt in raw_polygon],  # type: ignore[misc]
                color=_parse_color(entry.get("color")),
                alert_cooldown_seconds=float(
                    entry.get("alert_cooldown_seconds", _DEFAULT_COOLDOWN_SECONDS)
                ),
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_polygon(name: str, points: list[tuple[int, int]]) -> Polygon | None:
    """Attempt to build and validate a Shapely Polygon from *points*.

    Args:
        name: Zone name — used only for log messages.
        points: List of ``(x, y)`` integer pixel coordinates.

    Returns:
        A valid :class:`~shapely.geometry.Polygon`, or ``None`` if the geometry
        is degenerate (too few points, self-intersecting after repair, etc.).
    """
    if len(points) < _MIN_POLYGON_VERTICES:
        logger.warning(
            "Zone '{}': need at least {} vertices, got {}.",
            name, _MIN_POLYGON_VERTICES, len(points),
        )
        return None

    try:
        poly = Polygon(points)
    except Exception as exc:
        logger.error("Zone '{}': failed to construct polygon: {}", name, exc)
        return None

    if not poly.is_valid:
        logger.debug("Zone '{}': polygon not valid — attempting repair.", name)
        poly = make_valid(poly)
        # make_valid may return a MultiPolygon or GeometryCollection if the
        # input is severely degenerate; we only accept simple Polygons.
        if poly.geom_type != "Polygon":
            logger.warning(
                "Zone '{}': polygon could not be repaired to a simple Polygon "
                "(result: {}) — skipped.",
                name, poly.geom_type,
            )
            return None

    if poly.is_empty or poly.area == 0:
        logger.warning("Zone '{}': polygon has zero area — skipped.", name)
        return None

    return poly


def _parse_color(
    raw: Any,
    default: tuple[int, int, int] = _DEFAULT_ZONE_COLOR,
) -> tuple[int, int, int]:
    """Convert a raw color value from config/JSON into a BGR tuple.

    Args:
        raw: A list/tuple of three integers ``[B, G, R]`` or ``None``.
        default: Fallback color when *raw* is absent or malformed.

    Returns:
        A ``(B, G, R)`` integer tuple with values clamped to ``[0, 255]``.
    """
    if raw is None:
        return default
    try:
        b, g, r = int(raw[0]), int(raw[1]), int(raw[2])
        return (
            max(0, min(255, b)),
            max(0, min(255, g)),
            max(0, min(255, r)),
        )
    except (TypeError, IndexError, ValueError):
        logger.warning("Invalid color '{}' — using default.", raw)
        return default


def _resolve_probe(obj: TrackedObject) -> tuple[int, int]:
    """Return the best point to use for intrusion testing.

    Prefers :attr:`~src.tracking.tracker.TrackedObject.foot_point` (bottom-centre
    of the bounding box) as it approximates the floor contact point.  Falls back
    to :attr:`~src.tracking.tracker.TrackedObject.centroid` for legacy
    ``TrackedObject`` instances created without a ``foot_point``.

    Args:
        obj: A tracked person for the current frame.

    Returns:
        ``(x, y)`` pixel coordinate to test against zone polygons.
    """
    if obj.foot_point is not None:
        return obj.foot_point
    return obj.centroid


def _label_position(points: list[tuple[int, int]]) -> tuple[int, int]:
    """Compute a stable label anchor near the top-left of the polygon.

    Args:
        points: Polygon vertices as ``[(x, y), ...]``.

    Returns:
        ``(x, y)`` pixel coordinate for :func:`cv2.putText`.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys) - 6)
