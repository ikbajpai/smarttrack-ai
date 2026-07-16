"""
tracker.py
----------
ByteTrack multi-object tracker for SmartTrack AI.

Provides:
    TrackedObject  — frozen dataclass representing one person with a stable ID.
    Tracker        — wraps Ultralytics BYTETracker, maintains motion trails,
                     and returns structured output ready for ZoneManager.

This module depends only on the Detection output from detector.py; it never
imports ZoneManager, AlertManager, or Streamlit.  The pipeline layer wires
modules together.

Usage::

    import yaml
    import cv2
    from src.detection.detector import Detector
    from src.tracking.tracker import Tracker

    with open("config/config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    detector = Detector(cfg["model"])
    detector.load()

    tracker = Tracker(cfg["tracker"])

    for annotated_frame, detections in detector.stream_frames("video.mp4"):
        tracked = tracker.update(detections, annotated_frame)
        output = tracker.draw_annotations(annotated_frame, tracked)
        cv2.imshow("SmartTrack", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
"""

from __future__ import annotations

from argparse import Namespace
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger
from ultralytics.trackers.byte_tracker import BYTETracker

from src.detection.detector import Detection, detections_to_bytetrack_array

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TRAIL_LENGTH: int = 30
_DEFAULT_TRAIL_THICKNESS: int = 2
_DEFAULT_TRAIL_COLOUR: tuple[int, int, int] = (0, 255, 255)  # cyan BGR
_LABEL_TEXT_COLOUR: tuple[int, int, int] = (255, 255, 255)   # white
_HUD_COLOUR: tuple[int, int, int] = (0, 200, 255)            # amber


# ---------------------------------------------------------------------------
# TrackedObject dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackedObject:
    """An immutable record for one person with a stable cross-frame identity.

    Attributes:
        track_id: Unique, persistent integer assigned by ByteTrack.  Stable
            across frames for the lifetime of a track.
        bbox: Bounding box in pixel coordinates as ``(x1, y1, x2, y2)``.
        confidence: Detection confidence score in ``[0.0, 1.0]``.
        class_name: Always ``"person"`` for this pipeline.
        centroid: Pixel coordinates of the box centre ``(cx, cy)``.  Used by
            ZoneManager for point-in-polygon intrusion tests.
        foot_point: Bottom-centre of the bounding box ``(cx, y2)``.  Preferred
            contact point for floor-plane intrusion tests in ZoneManager because
            it approximates where the person stands.  ``None`` only when created
            without a valid bbox (e.g., in legacy unit-test fixtures).
    """

    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
    centroid: tuple[int, int]
    foot_point: tuple[int, int] | None = None


# ---------------------------------------------------------------------------
# Internal BYTETracker adapter
# ---------------------------------------------------------------------------

class _BoxProxy:
    """Minimal proxy that satisfies the BYTETracker ``results`` interface.

    BYTETracker._split_detections() accesses ``.conf`` and indexes the proxy
    with boolean masks (``results[mask]``).  BYTETracker.init_track() calls
    ``parse_bboxes(results)`` which reads ``.xywh`` (center format: cx, cy, w,
    h) and then indexes ``.conf`` and ``.cls`` element-wise.

    All attributes are plain NumPy arrays; boolean mask indexing is forwarded
    to each array simultaneously via ``__getitem__``.

    Args:
        xywh: Float32 array of shape ``(N, 4)`` — center-format bounding boxes
              ``[cx, cy, w, h]``.
        conf: Float32 array of shape ``(N,)``   — confidence scores.
        cls:  Float32 array of shape ``(N,)``   — class IDs.
    """

    def __init__(
        self,
        xywh: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __getitem__(self, mask: np.ndarray) -> _BoxProxy:
        """Return a new proxy containing only the rows selected by *mask*.

        Args:
            mask: Boolean or integer index array.

        Returns:
            A new :class:`_BoxProxy` with the selected rows.
        """
        return _BoxProxy(self.xywh[mask], self.conf[mask], self.cls[mask])

    def __len__(self) -> int:
        """Return the number of detections in this proxy."""
        return len(self.conf)


# ---------------------------------------------------------------------------
# Tracker class
# ---------------------------------------------------------------------------

class Tracker:
    """ByteTrack wrapper with stable track IDs and configurable motion trails.

    Wraps Ultralytics ``BYTETracker``, accepts :class:`Detection` objects
    from the detector, and returns :class:`TrackedObject` instances with
    persistent IDs for consumption by ZoneManager.

    Args:
        config: The ``tracker`` section of ``config/config.yaml``.  Expected
            keys::

                track_high_thresh: 0.5
                track_low_thresh:  0.1
                new_track_thresh:  0.6
                track_buffer:      30
                match_thresh:      0.8
                frame_rate:        30    # should match video.fps
                trail_max_length:  30
                trail_thickness:   2
                trail_colour:      [0, 255, 255]   # BGR list

    Example::

        import yaml
        from src.tracking.tracker import Tracker

        with open("config/config.yaml") as fh:
            cfg = yaml.safe_load(fh)

        tracker = Tracker(cfg["tracker"])
        tracked = tracker.update(detections, frame)
    """

    def __init__(self, config: dict) -> None:
        self._track_high_thresh: float = float(config.get("track_high_thresh", 0.5))
        self._track_low_thresh: float = float(config.get("track_low_thresh", 0.1))
        self._new_track_thresh: float = float(config.get("new_track_thresh", 0.6))
        self._track_buffer: int = int(config.get("track_buffer", 30))
        self._match_thresh: float = float(config.get("match_thresh", 0.8))
        self._fuse_score: bool = bool(config.get("fuse_score", False))
        self._frame_rate: int = int(config.get("frame_rate", 30))

        self._trail_length: int = int(config.get("trail_max_length", _DEFAULT_TRAIL_LENGTH))
        self._trail_thickness: int = int(config.get("trail_thickness", _DEFAULT_TRAIL_THICKNESS))
        raw_colour = config.get("trail_colour", list(_DEFAULT_TRAIL_COLOUR))
        self._trail_colour: tuple[int, int, int] = (
            int(raw_colour[0]), int(raw_colour[1]), int(raw_colour[2])
        )

        # track_id → centroid history for trail drawing
        self._trails: dict[int, deque[tuple[int, int]]] = {}

        self._byte_tracker: BYTETracker = self._build_byte_tracker()

        logger.debug(
            "Tracker configured | high={} low={} new={} buffer={} match={} "
            "trail_len={} frame_rate={}",
            self._track_high_thresh,
            self._track_low_thresh,
            self._new_track_thresh,
            self._track_buffer,
            self._match_thresh,
            self._trail_length,
            self._frame_rate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> list[TrackedObject]:
        """Update tracker state with new per-frame detections.

        Converts :class:`Detection` objects to the format expected by
        BYTETracker, runs the tracker update, records centroid history for
        trail drawing, and returns structured :class:`TrackedObject` results.

        Args:
            detections: Output of :meth:`~src.detection.detector.Detector.detect`
                for the current frame.
            frame: The BGR frame that produced *detections*.  Passed to
                BYTETracker for image-space Kalman predictions.

        Returns:
            List of :class:`TrackedObject`, one per active track.  Returns an
            empty list when *frame* is invalid or ByteTrack produces no tracks.
        """
        if frame is None or frame.size == 0:
            logger.warning("update() received an invalid frame — skipping.")
            return []

        proxy = self._build_proxy(detections)

        try:
            raw_output = self._byte_tracker.update(proxy, frame)
        except Exception as exc:
            logger.error("BYTETracker update error: {}", exc)
            return []

        # _format_output() returns shape (N, 8): [x1, y1, x2, y2, track_id, score, cls, idx]
        # Returns an empty array (shape (0,) or (0, 8)) when no tracks are active.
        tracked: list[TrackedObject] = []
        if raw_output is None or len(raw_output) == 0:
            logger.debug("0 detection(s) in → 0 active track(s) out.")
            return []

        for row in raw_output:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            track_id = int(row[4])
            score = float(row[5])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if track_id not in self._trails:
                self._trails[track_id] = deque(maxlen=self._trail_length)
            self._trails[track_id].append((cx, cy))

            tracked.append(
                TrackedObject(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                    class_name="person",
                    centroid=(cx, cy),
                    foot_point=(cx, y2),
                )
            )

        logger.debug(
            "{} detection(s) in → {} active track(s) out.",
            len(detections),
            len(tracked),
        )
        return tracked

    def draw_annotations(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        *,
        show_trails: bool = True,
        show_ids: bool = True,
        show_count: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes, track IDs, and motion trails onto a copy of *frame*.

        The source *frame* is never mutated; a copy is returned.

        Args:
            frame: BGR frame to annotate.
            tracked_objects: Output of :meth:`update` for this frame.
            show_trails: Render faded centroid trails for each active track.
            show_ids: Render track ID labels above each bounding box.
            show_count: Render active track count in the top-left HUD.

        Returns:
            A new BGR frame with all annotations applied.
        """
        annotated = frame.copy()

        if show_trails:
            active_ids = {obj.track_id for obj in tracked_objects}
            self._draw_trails(annotated, active_ids)

        for obj in tracked_objects:
            self._draw_box(annotated, obj, show_id=show_ids)

        if show_count:
            cv2.putText(
                annotated,
                f"Tracked: {len(tracked_objects)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                _HUD_COLOUR,
                2,
                cv2.LINE_AA,
            )

        return annotated

    def reset(self) -> None:
        """Reset tracker state between video sources.

        Clears all trail history and re-initialises BYTETracker so that track
        IDs restart from 1 and there is no carry-over between clips.
        """
        self._trails.clear()
        self._byte_tracker = self._build_byte_tracker()
        logger.info("Tracker reset — trails cleared, BYTETracker re-initialised.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_trail_count(self) -> int:
        """Number of track IDs that currently have recorded trail history."""
        return len(self._trails)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_byte_tracker(self) -> BYTETracker:
        """Construct a fresh BYTETracker from stored hyper-parameters.

        Returns:
            A new :class:`BYTETracker` instance.
        """
        args = Namespace(
            track_high_thresh=self._track_high_thresh,
            track_low_thresh=self._track_low_thresh,
            new_track_thresh=self._new_track_thresh,
            track_buffer=self._track_buffer,
            match_thresh=self._match_thresh,
            fuse_score=self._fuse_score,
            mot20=False,
        )
        bt = BYTETracker(args)
        logger.debug("BYTETracker instance created.")
        return bt

    def _build_proxy(self, detections: list[Detection]) -> _BoxProxy:
        """Convert detections to a BYTETracker-compatible proxy object.

        BYTETracker expects bounding boxes in center format ``[cx, cy, w, h]``
        (accessed via ``.xywh``).  The raw detection array from
        :func:`~src.detection.detector.detections_to_bytetrack_array` uses
        ``[x1, y1, x2, y2]`` (xyxy), so a conversion is performed here.

        Args:
            detections: Detector output for a single frame.

        Returns:
            A :class:`_BoxProxy` whose ``.xywh``, ``.conf``, and ``.cls``
            attributes are NumPy arrays sized for the number of detections,
            ready for BYTETracker consumption.
        """
        arr = detections_to_bytetrack_array(detections)  # (N, 6) float32

        if arr.shape[0] == 0:
            return _BoxProxy(
                xywh=np.empty((0, 4), dtype=np.float32),
                conf=np.empty(0, dtype=np.float32),
                cls=np.empty(0, dtype=np.float32),
            )

        # Convert xyxy → center xywh: [cx, cy, w, h]
        xyxy = arr[:, :4]
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)

        return _BoxProxy(
            xywh=xywh,
            conf=arr[:, 4],
            cls=arr[:, 5],
        )

    def _draw_trails(
        self,
        frame: np.ndarray,
        active_ids: set[int],
    ) -> None:
        """Draw faded centroid trails for active track IDs in-place.

        Points closer to the present are brighter; points at the tail of the
        deque (oldest) are the darkest.  The line thickness is taken from
        ``tracker.trail_thickness`` in config.

        Args:
            frame: BGR frame to annotate (mutated directly).
            active_ids: Set of track IDs currently returned by BYTETracker.
                Only tracks in this set have their trails rendered so that
                stale history from lost tracks is not shown.
        """
        for track_id in active_ids:
            trail = self._trails.get(track_id)
            if trail is None or len(trail) < 2:
                continue

            colour = self._track_colour(track_id)
            points = list(trail)
            n = len(points)

            for i in range(1, n):
                # Linearly fade older segments toward black.
                alpha = i / n
                faded = (
                    int(colour[0] * alpha),
                    int(colour[1] * alpha),
                    int(colour[2] * alpha),
                )
                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    faded,
                    self._trail_thickness,
                    cv2.LINE_AA,
                )

    def _draw_box(
        self,
        frame: np.ndarray,
        obj: TrackedObject,
        *,
        show_id: bool,
    ) -> None:
        """Draw a bounding box and optional track ID label in-place.

        The box colour is deterministically derived from the track ID so that
        the same person always has the same colour across frames.

        Args:
            frame: BGR frame to annotate (mutated directly).
            obj: Tracked object to render.
            show_id: Whether to render the ID label above the box.
        """
        x1, y1, x2, y2 = obj.bbox
        colour = self._track_colour(obj.track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        if show_id:
            label = f"ID:{obj.track_id}  {obj.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_top = max(y1 - text_h - baseline - 4, 0)
            cv2.rectangle(
                frame,
                (x1, label_top),
                (x1 + text_w, label_top + text_h + baseline + 4),
                colour,
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                label,
                (x1, label_top + text_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                _LABEL_TEXT_COLOUR,
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _track_colour(track_id: int) -> tuple[int, int, int]:
        """Return a stable, perceptually distinct BGR colour for *track_id*.

        Uses the golden-angle trick in HSV space: multiplying by 137 (the
        integer approximation of 360° × golden ratio) spreads IDs across the
        hue wheel without clustering.

        Args:
            track_id: The stable ByteTrack integer ID.

        Returns:
            A ``(B, G, R)`` tuple suitable for OpenCV drawing functions.
        """
        hue = int(track_id * 137) % 180  # 180-degree HSV hue range in OpenCV
        hsv_pixel = np.uint8([[[hue, 220, 220]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))
