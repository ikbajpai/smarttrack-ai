"""
test_inference_pipeline.py
--------------------------
Unit tests for InferencePipeline and PipelineResult.

All four downstream modules (Detector, Tracker, ZoneManager, AlertManager) are
replaced with MagicMock instances so that no YOLO weights, video files, or
external services are required.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.alerts.alert_manager import AlertManager
from src.detection.detector import Detection, Detector
from src.pipeline.inference_pipeline import InferencePipeline, PipelineResult
from src.tracking.tracker import TrackedObject, Tracker
from src.zones.zone_manager import EventType, IntrusionEvent, ZoneManager

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

BLANK_FRAME: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)
NONBLANK_FRAME: np.ndarray = np.ones((480, 640, 3), dtype=np.uint8) * 128


def _detection(track_id: int = 1) -> Detection:
    return Detection(bbox=(10, 10, 100, 200), confidence=0.9, class_id=0, class_name="person")


def _tracked(track_id: int = 1) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=(10, 10, 100, 200),
        confidence=0.9,
        class_name="person",
        centroid=(55, 105),
        foot_point=(55, 200),
    )


def _event(track_id: int = 1) -> IntrusionEvent:
    return IntrusionEvent(
        timestamp=datetime.utcnow(),
        track_id=track_id,
        zone_name="Zone A",
        event_type=EventType.ENTER,
    )


# ---------------------------------------------------------------------------
# Fixture: fully mocked pipeline
# ---------------------------------------------------------------------------

@pytest.fixture()
def mocks() -> dict:
    """Return four MagicMock module instances with sensible default return values."""
    detector = MagicMock(spec=Detector)
    tracker = MagicMock(spec=Tracker)
    zone_manager = MagicMock(spec=ZoneManager)
    alert_manager = MagicMock(spec=AlertManager)

    detector.detect.return_value = []
    tracker.update.return_value = []
    tracker.draw_annotations.return_value = BLANK_FRAME.copy()
    zone_manager.check_intrusions.return_value = []
    zone_manager.draw_zones.return_value = BLANK_FRAME.copy()
    alert_manager.process.return_value = 0

    return {
        "detector": detector,
        "tracker": tracker,
        "zone_manager": zone_manager,
        "alert_manager": alert_manager,
    }


@pytest.fixture()
def pipeline(mocks: dict) -> InferencePipeline:
    return InferencePipeline(**mocks)


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def test_fields_accessible(self):
        result = PipelineResult(
            annotated_frame=BLANK_FRAME.copy(),
            tracked_objects=[],
            intrusion_events=[],
            people_count=0,
            active_track_count=0,
            fps=30.0,
            processing_time_ms=5.0,
        )
        assert result.people_count == 0
        assert result.fps == 30.0
        assert result.processing_time_ms == 5.0
        assert result.tracked_objects == []
        assert result.intrusion_events == []

    def test_frozen_prevents_reassignment(self):
        result = PipelineResult(
            annotated_frame=BLANK_FRAME.copy(),
            tracked_objects=[],
            intrusion_events=[],
            people_count=0,
            active_track_count=0,
            fps=0.0,
            processing_time_ms=0.0,
        )
        with pytest.raises(Exception):
            result.fps = 99.0  # type: ignore[misc]

    def test_annotated_frame_is_ndarray(self):
        result = PipelineResult(
            annotated_frame=BLANK_FRAME.copy(),
            tracked_objects=[],
            intrusion_events=[],
            people_count=3,
            active_track_count=3,
            fps=25.0,
            processing_time_ms=10.0,
        )
        assert isinstance(result.annotated_frame, np.ndarray)


# ---------------------------------------------------------------------------
# InferencePipeline construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_pipeline_created(self, pipeline: InferencePipeline):
        assert pipeline is not None

    def test_fps_zero_before_first_frame(self, pipeline: InferencePipeline):
        assert pipeline.current_fps == 0.0

    def test_frames_processed_zero_before_first_frame(self, pipeline: InferencePipeline):
        assert pipeline.frames_processed == 0

    def test_stores_injected_modules(self, pipeline: InferencePipeline, mocks: dict):
        assert pipeline._detector is mocks["detector"]
        assert pipeline._tracker is mocks["tracker"]
        assert pipeline._zone_manager is mocks["zone_manager"]
        assert pipeline._alert_manager is mocks["alert_manager"]


# ---------------------------------------------------------------------------
# process_frame — call order and wiring
# ---------------------------------------------------------------------------

class TestProcessFrameCallOrder:
    def test_detector_called_with_frame(self, pipeline: InferencePipeline, mocks: dict):
        pipeline.process_frame(BLANK_FRAME)
        mocks["detector"].detect.assert_called_once_with(BLANK_FRAME)

    def test_tracker_called_with_detections_and_frame(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        detections = [_detection()]
        mocks["detector"].detect.return_value = detections
        pipeline.process_frame(BLANK_FRAME)
        mocks["tracker"].update.assert_called_once_with(detections, BLANK_FRAME)

    def test_zone_manager_called_with_tracked_objects(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        tracked = [_tracked()]
        mocks["tracker"].update.return_value = tracked
        pipeline.process_frame(BLANK_FRAME)
        mocks["zone_manager"].check_intrusions.assert_called_once_with(tracked)

    def test_alert_manager_called_with_events(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        events = [_event()]
        mocks["zone_manager"].check_intrusions.return_value = events
        pipeline.process_frame(BLANK_FRAME)
        mocks["alert_manager"].process.assert_called_once_with(events)

    def test_tracker_draw_annotations_called(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        tracked = [_tracked()]
        mocks["tracker"].update.return_value = tracked
        pipeline.process_frame(BLANK_FRAME)
        mocks["tracker"].draw_annotations.assert_called_once_with(BLANK_FRAME, tracked)

    def test_zone_draw_zones_called_on_tracker_output(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        tracker_annotated = NONBLANK_FRAME.copy()
        mocks["tracker"].draw_annotations.return_value = tracker_annotated
        pipeline.process_frame(BLANK_FRAME)
        mocks["zone_manager"].draw_zones.assert_called_once_with(tracker_annotated)


# ---------------------------------------------------------------------------
# process_frame — PipelineResult contents
# ---------------------------------------------------------------------------

class TestProcessFrameResult:
    def test_returns_pipeline_result(self, pipeline: InferencePipeline):
        result = pipeline.process_frame(BLANK_FRAME)
        assert isinstance(result, PipelineResult)

    def test_people_count_from_detections(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        mocks["detector"].detect.return_value = [_detection(), _detection()]
        result = pipeline.process_frame(BLANK_FRAME)
        assert result.people_count == 2

    def test_active_track_count_from_tracker(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        mocks["tracker"].update.return_value = [_tracked(1), _tracked(2), _tracked(3)]
        result = pipeline.process_frame(BLANK_FRAME)
        assert result.active_track_count == 3

    def test_intrusion_events_from_zone_manager(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        events = [_event(1), _event(2)]
        mocks["zone_manager"].check_intrusions.return_value = events
        result = pipeline.process_frame(BLANK_FRAME)
        assert result.intrusion_events == events

    def test_tracked_objects_in_result(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        tracked = [_tracked(5)]
        mocks["tracker"].update.return_value = tracked
        result = pipeline.process_frame(BLANK_FRAME)
        assert result.tracked_objects == tracked

    def test_annotated_frame_is_zone_manager_output(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        zone_output = NONBLANK_FRAME.copy()
        mocks["zone_manager"].draw_zones.return_value = zone_output
        result = pipeline.process_frame(BLANK_FRAME)
        np.testing.assert_array_equal(result.annotated_frame, zone_output)

    def test_processing_time_ms_positive(self, pipeline: InferencePipeline):
        result = pipeline.process_frame(BLANK_FRAME)
        assert result.processing_time_ms > 0.0

    def test_fps_updates_after_first_frame(self, pipeline: InferencePipeline):
        pipeline.process_frame(BLANK_FRAME)
        assert pipeline.current_fps > 0.0

    def test_frames_processed_increments(self, pipeline: InferencePipeline):
        pipeline.process_frame(BLANK_FRAME)
        pipeline.process_frame(BLANK_FRAME)
        assert pipeline.frames_processed == 2


# ---------------------------------------------------------------------------
# process_frame — invalid frame handling
# ---------------------------------------------------------------------------

class TestInvalidFrame:
    def test_none_frame_returns_result(self, pipeline: InferencePipeline):
        result = pipeline.process_frame(None)  # type: ignore[arg-type]
        assert isinstance(result, PipelineResult)

    def test_none_frame_annotated_frame_is_ndarray(self, pipeline: InferencePipeline):
        result = pipeline.process_frame(None)  # type: ignore[arg-type]
        assert isinstance(result.annotated_frame, np.ndarray)

    def test_none_frame_counts_zero(self, pipeline: InferencePipeline):
        result = pipeline.process_frame(None)  # type: ignore[arg-type]
        assert result.people_count == 0
        assert result.active_track_count == 0

    def test_empty_array_frame_returns_result(self, pipeline: InferencePipeline):
        result = pipeline.process_frame(np.empty(0, dtype=np.uint8))
        assert isinstance(result, PipelineResult)
        assert result.people_count == 0

    def test_none_frame_no_module_calls(self, pipeline: InferencePipeline, mocks: dict):
        pipeline.process_frame(None)  # type: ignore[arg-type]
        mocks["detector"].detect.assert_not_called()
        mocks["tracker"].update.assert_not_called()


# ---------------------------------------------------------------------------
# process_frame — stage failure isolation
# ---------------------------------------------------------------------------

class TestStageFaultIsolation:
    def test_detector_failure_returns_result(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        mocks["detector"].detect.side_effect = RuntimeError("model crash")
        result = pipeline.process_frame(BLANK_FRAME)
        assert isinstance(result, PipelineResult)
        assert result.people_count == 0

    def test_detector_failure_tracker_still_called(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        mocks["detector"].detect.side_effect = RuntimeError("model crash")
        pipeline.process_frame(BLANK_FRAME)
        # tracker is called with empty detections list
        mocks["tracker"].update.assert_called_once_with([], BLANK_FRAME)

    def test_tracker_failure_returns_result(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        mocks["tracker"].update.side_effect = RuntimeError("tracker crash")
        result = pipeline.process_frame(BLANK_FRAME)
        assert isinstance(result, PipelineResult)
        assert result.active_track_count == 0

    def test_tracker_failure_zone_manager_still_called(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        mocks["tracker"].update.side_effect = RuntimeError("tracker crash")
        pipeline.process_frame(BLANK_FRAME)
        mocks["zone_manager"].check_intrusions.assert_called_once_with([])

    def test_zone_manager_failure_returns_result(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        mocks["zone_manager"].check_intrusions.side_effect = RuntimeError("zone crash")
        result = pipeline.process_frame(BLANK_FRAME)
        assert isinstance(result, PipelineResult)
        assert result.intrusion_events == []

    def test_zone_manager_failure_alert_manager_still_called(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        mocks["zone_manager"].check_intrusions.side_effect = RuntimeError("zone crash")
        pipeline.process_frame(BLANK_FRAME)
        mocks["alert_manager"].process.assert_called_once_with([])

    def test_alert_manager_failure_does_not_crash(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        mocks["alert_manager"].process.side_effect = RuntimeError("alert crash")
        result = pipeline.process_frame(BLANK_FRAME)
        assert isinstance(result, PipelineResult)

    def test_tracker_annotation_failure_returns_frame_copy(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        mocks["tracker"].draw_annotations.side_effect = RuntimeError("draw crash")
        result = pipeline.process_frame(NONBLANK_FRAME)
        assert isinstance(result.annotated_frame, np.ndarray)


# ---------------------------------------------------------------------------
# reset and teardown
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_calls_tracker_reset(self, pipeline: InferencePipeline, mocks: dict):
        pipeline.reset()
        mocks["tracker"].reset.assert_called_once()

    def test_reset_calls_zone_manager_reset(self, pipeline: InferencePipeline, mocks: dict):
        pipeline.reset()
        mocks["zone_manager"].reset.assert_called_once()

    def test_reset_calls_alert_manager_clear_history(
        self, pipeline: InferencePipeline, mocks: dict
    ):
        pipeline.reset()
        mocks["alert_manager"].clear_history.assert_called_once()

    def test_reset_clears_fps_window(self, pipeline: InferencePipeline):
        pipeline.process_frame(BLANK_FRAME)
        pipeline.reset()
        assert pipeline.current_fps == 0.0
        assert pipeline.frames_processed == 0

    def test_teardown_calls_reset(self, pipeline: InferencePipeline, mocks: dict):
        pipeline.teardown()
        mocks["tracker"].reset.assert_called_once()
        mocks["zone_manager"].reset.assert_called_once()
