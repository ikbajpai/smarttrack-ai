"""
inference_pipeline.py
---------------------
Central orchestration layer for SmartTrack AI.

Provides:
    PipelineResult   — Frozen dataclass returned by every process_frame() call.
    InferencePipeline — Wires Detector → Tracker → ZoneManager → AlertManager
                        in sequence and returns a structured result.

This module contains no business logic.  Detection thresholds, polygon
geometry, and alert cooldowns live inside their respective modules.
InferencePipeline only calls those modules in order and packages their output.

Dependency injection is used throughout: all four module instances are passed
into the constructor rather than constructed internally.  This makes it trivial
to substitute mocks in tests or swap implementations without modifying the
pipeline.

Usage::

    import yaml
    import cv2
    from src.detection.detector import Detector
    from src.tracking.tracker import Tracker
    from src.zones.zone_manager import ZoneManager
    from src.alerts.alert_manager import AlertManager
    from src.pipeline.inference_pipeline import InferencePipeline

    with open("config/config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    detector = Detector(cfg["model"])
    detector.load()

    pipeline = InferencePipeline(
        detector=detector,
        tracker=Tracker(cfg["tracker"]),
        zone_manager=ZoneManager(cfg["zones"]),
        alert_manager=AlertManager(cfg["alerts"]),
    )

    for frame in frames:
        result = pipeline.process_frame(frame)
        cv2.imshow("SmartTrack", result.annotated_frame)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from src.alerts.alert_manager import AlertManager
from src.detection.detector import Detection, Detector
from src.tracking.tracker import TrackedObject, Tracker
from src.zones.zone_manager import IntrusionEvent, ZoneManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FPS_WINDOW: int = 30   # frames averaged for smoothed FPS
_EMPTY_FRAME_SHAPE: tuple[int, int, int] = (480, 640, 3)


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class PipelineResult:
    """Immutable snapshot of one processed frame and all pipeline outputs.

    ``frozen=True`` prevents accidental attribute reassignment after creation.
    ``eq=False`` suppresses auto-generated ``__eq__`` / ``__hash__`` so that
    containing a :class:`numpy.ndarray` does not raise at comparison or hash
    time.

    Attributes:
        annotated_frame: BGR frame with tracker boxes, motion trails, and zone
            overlays composited in that order.  Always a valid ndarray — a
            black placeholder is used when the input frame was invalid.
        tracked_objects: Active tracks returned by the Tracker for this frame.
        intrusion_events: ENTER / EXIT events returned by ZoneManager.
        people_count: Number of persons detected (raw detector output count).
        active_track_count: Number of stable ByteTrack IDs in this frame.
        fps: Smoothed pipeline throughput over the last ``_FPS_WINDOW`` frames.
        processing_time_ms: Wall-clock time for this single frame in milliseconds.
    """

    annotated_frame: np.ndarray
    tracked_objects: list[TrackedObject]
    intrusion_events: list[IntrusionEvent]
    people_count: int
    active_track_count: int
    fps: float
    processing_time_ms: float


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """Orchestrates the full Detector → Tracker → ZoneManager → AlertManager pipeline.

    Each call to :meth:`process_frame` runs exactly one frame through all four
    modules in sequence and returns a :class:`PipelineResult`.  No internal
    module state is exposed on the pipeline object itself.

    Module failures are isolated per stage: if any module raises an exception,
    that stage returns an empty result (empty list / original frame) and
    processing continues with the next stage.  The pipeline never crashes the
    calling application due to a downstream module error.

    Args:
        detector: Loaded :class:`~src.detection.detector.Detector` instance.
            :meth:`~src.detection.detector.Detector.load` must have been called
            before passing it here.
        tracker: Configured :class:`~src.tracking.tracker.Tracker` instance.
        zone_manager: Configured :class:`~src.zones.zone_manager.ZoneManager`
            instance with zones already registered.
        alert_manager: Configured :class:`~src.alerts.alert_manager.AlertManager`
            instance with handlers registered.

    Example::

        pipeline = InferencePipeline(
            detector=detector,
            tracker=Tracker(cfg["tracker"]),
            zone_manager=ZoneManager(cfg["zones"]),
            alert_manager=AlertManager(cfg["alerts"]),
        )

        result = pipeline.process_frame(frame)
        print(result.people_count, result.fps)
    """

    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        zone_manager: ZoneManager,
        alert_manager: AlertManager,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._zone_manager = zone_manager
        self._alert_manager = alert_manager

        # Rolling window of per-frame wall-clock durations (seconds).
        self._frame_times: deque[float] = deque(maxlen=_FPS_WINDOW)

        logger.info(
            "InferencePipeline ready | detector={} tracker={} zone_manager={} alert_manager={}",
            type(detector).__name__,
            type(tracker).__name__,
            type(zone_manager).__name__,
            type(alert_manager).__name__,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """Run the full pipeline on a single BGR video frame.

        Stages in order:

        1. **Validate** — reject None / empty frames immediately.
        2. **Detect** — :meth:`~src.detection.detector.Detector.detect`.
        3. **Track** — :meth:`~src.tracking.tracker.Tracker.update`.
        4. **Zone check** — :meth:`~src.zones.zone_manager.ZoneManager.check_intrusions`.
        5. **Alert** — :meth:`~src.alerts.alert_manager.AlertManager.process`.
        6. **Annotate** — tracker boxes + trails, then zone overlays.
        7. **Return** :class:`PipelineResult`.

        Each stage is individually guarded; an exception in one stage causes
        that stage to return an empty result and processing continues.

        Args:
            frame: BGR image as returned by ``cv2.VideoCapture.read()``.

        Returns:
            A :class:`PipelineResult` for this frame.  ``annotated_frame`` is
            always a valid ndarray.  All list fields default to empty on error.
        """
        t0 = time.perf_counter()

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("process_frame() received an invalid frame — returning empty result.")
            return self._empty_result(t0)

        # ── Stage 1: Detect ───────────────────────────────────────────
        detections: list[Detection] = []
        try:
            detections = self._detector.detect(frame)
        except Exception as exc:
            logger.error("Detector stage failed: {}", exc)

        # ── Stage 2: Track ────────────────────────────────────────────
        tracked_objects: list[TrackedObject] = []
        try:
            tracked_objects = self._tracker.update(detections, frame)
        except Exception as exc:
            logger.error("Tracker stage failed: {}", exc)

        # ── Stage 3: Zone check ───────────────────────────────────────
        intrusion_events: list[IntrusionEvent] = []
        try:
            intrusion_events = self._zone_manager.check_intrusions(tracked_objects)
        except Exception as exc:
            logger.error("ZoneManager stage failed: {}", exc)

        # ── Stage 4: Alert ────────────────────────────────────────────
        try:
            self._alert_manager.process(intrusion_events)
        except Exception as exc:
            logger.error("AlertManager stage failed: {}", exc)

        # ── Stage 5: Annotate ─────────────────────────────────────────
        annotated = self._annotate(frame, tracked_objects)

        # ── Timing ───────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        self._frame_times.append(elapsed)

        result = PipelineResult(
            annotated_frame=annotated,
            tracked_objects=tracked_objects,
            intrusion_events=intrusion_events,
            people_count=len(detections),
            active_track_count=len(tracked_objects),
            fps=self._current_fps(),
            processing_time_ms=elapsed * 1_000,
        )

        logger.debug(
            "Frame processed | people={} tracks={} events={} {:.1f}ms fps={:.1f}",
            result.people_count,
            result.active_track_count,
            len(result.intrusion_events),
            result.processing_time_ms,
            result.fps,
        )

        return result

    def reset(self) -> None:
        """Reset stateful modules between video sources.

        Clears tracker trails and re-initialises BYTETracker, and clears
        ZoneManager's active-intrusion state so that stale pairs from the
        previous clip do not trigger spurious EXIT events.  AlertManager
        history and pipeline FPS history are also cleared.

        Call this whenever switching to a new video file or webcam session.
        """
        self._tracker.reset()
        self._zone_manager.reset()
        self._alert_manager.clear_history()
        self._frame_times.clear()
        logger.info("InferencePipeline reset for new source.")

    def teardown(self) -> None:
        """Release pipeline resources.

        Calls :meth:`reset` to flush all stateful module state.  Extend this
        method in a subclass if future modules acquire file handles, network
        sockets, or GPU contexts that need explicit release.
        """
        self.reset()
        logger.info("InferencePipeline torn down.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_fps(self) -> float:
        """Smoothed pipeline throughput over the last ``_FPS_WINDOW`` frames.

        Returns ``0.0`` until at least one frame has been processed.
        """
        return self._current_fps()

    @property
    def frames_processed(self) -> int:
        """Number of frames in the current FPS window (up to ``_FPS_WINDOW``)."""
        return len(self._frame_times)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _annotate(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
    ) -> np.ndarray:
        """Composite tracker annotations then zone overlays onto *frame*.

        The source *frame* is never mutated.  Both annotation methods produce
        copies internally, so only one copy is made here overall.

        Args:
            frame: Original BGR frame.
            tracked_objects: Active tracks for this frame.

        Returns:
            Annotated BGR frame with tracker boxes, trails, and zone polygons.
        """
        try:
            annotated = self._tracker.draw_annotations(frame, tracked_objects)
        except Exception as exc:
            logger.error("Tracker annotation failed: {}", exc)
            annotated = frame.copy()

        try:
            annotated = self._zone_manager.draw_zones(annotated)
        except Exception as exc:
            logger.error("ZoneManager annotation failed: {}", exc)

        return annotated

    def _current_fps(self) -> float:
        """Compute smoothed FPS from the rolling frame-time window.

        Returns:
            Frames per second, or ``0.0`` when no frames have been processed.
        """
        if not self._frame_times:
            return 0.0
        avg = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg if avg > 0.0 else 0.0

    def _empty_result(self, t0: float) -> PipelineResult:
        """Return a valid but empty :class:`PipelineResult` for invalid frames.

        Args:
            t0: :func:`time.perf_counter` value recorded at the start of
                :meth:`process_frame`, used to compute ``processing_time_ms``.

        Returns:
            A :class:`PipelineResult` with a black placeholder frame and all
            counts set to zero.
        """
        elapsed = time.perf_counter() - t0
        return PipelineResult(
            annotated_frame=np.zeros(_EMPTY_FRAME_SHAPE, dtype=np.uint8),
            tracked_objects=[],
            intrusion_events=[],
            people_count=0,
            active_track_count=0,
            fps=self._current_fps(),
            processing_time_ms=elapsed * 1_000,
        )
