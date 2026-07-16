"""
streamlit_app.py
----------------
SmartTrack AI — Streamlit web application.

Thin presentation layer that interacts exclusively with InferencePipeline.
No detection, tracking, zone, or alert business logic lives here.

Architecture
------------
- One frame is processed per Streamlit script execution.
- ``st.rerun()`` is called after each frame so the Stop button and sidebar
  controls remain responsive between frames.
- The pipeline (including YOLO weights) is loaded once via
  ``@st.cache_resource`` and reused across all reruns.
- All per-user, per-session state lives in ``st.session_state``.

Run
---
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import yaml

from src.alerts.alert_manager import AlertManager
from src.detection.detector import Detector
from src.pipeline.inference_pipeline import InferencePipeline, PipelineResult
from src.tracking.tracker import Tracker
from src.zones.zone_manager import EventType, ZoneManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_PATH: str = "config/config.yaml"
_DEFAULT_ZONE_JSON: str = "data/zone_configs/default_zones.json"
_CONFIDENCE_MIN: float = 0.1
_CONFIDENCE_MAX: float = 0.9
_CONFIDENCE_STEP: float = 0.05
_MAX_ALERT_ROWS: int = 100
_FRAME_DELAY_S: float = 1.0 / 30  # ~30 FPS target between frames


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_config() -> dict[str, Any]:
    """Load and cache ``config/config.yaml``.

    Returns:
        Parsed configuration dictionary.  Cached for the lifetime of the
        Streamlit server process.
    """
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Pipeline construction — loaded once, shared across reruns
# ---------------------------------------------------------------------------

@st.cache_resource
def _build_pipeline(_config_path: str) -> tuple[InferencePipeline, Detector]:
    """Construct and cache the full InferencePipeline.

    Decorated with ``@st.cache_resource`` so YOLO weights are loaded only
    once per server process, regardless of how many times the Streamlit
    script reruns.

    Args:
        _config_path: Path to ``config.yaml``.  Used as the cache key.
            The leading underscore tells Streamlit not to try to hash its
            value (the file contents are read inside the function).

    Returns:
        ``(pipeline, detector)`` — the :class:`Detector` is returned
        separately so the sidebar confidence slider can mutate
        ``detector.confidence_threshold`` at runtime without going through
        the pipeline.
    """
    with open(_config_path) as fh:
        cfg = yaml.safe_load(fh)

    detector = Detector(cfg["model"])
    detector.load()

    tracker = Tracker(cfg["tracker"])

    zone_manager = ZoneManager(cfg["zones"])
    default_zones = Path(_DEFAULT_ZONE_JSON)
    if default_zones.exists():
        zone_manager.load_zones_from_file(default_zones)

    alert_manager = AlertManager(cfg["alerts"])

    pipeline = InferencePipeline(
        detector=detector,
        tracker=tracker,
        zone_manager=zone_manager,
        alert_manager=alert_manager,
    )

    return pipeline, detector


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialise ``st.session_state`` keys that must persist across reruns.

    Safe to call on every script execution — keys are only set when absent.
    """
    defaults: dict[str, Any] = {
        "running": False,
        "cap": None,                # cv2.VideoCapture | None
        "source": None,             # int (webcam index) or str (file path)
        "temp_file_path": None,     # path to uploaded temp file for cleanup
        "alert_log": [],            # list[dict] for the alert table
        "last_result": None,        # PipelineResult | None
        "total_intrusions": 0,      # cumulative ENTER events this session
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(detector: Detector) -> tuple[Any, int | str | None]:
    """Render the sidebar controls and return the selected video source.

    Args:
        detector: Live detector whose confidence threshold can be updated
            at runtime via the slider.

    Returns:
        ``(uploaded_file, webcam_index)`` — only one will be non-None based
        on the user's selection.  Returns ``(None, None)`` when neither is
        configured yet.
    """
    cfg = _load_config()

    with st.sidebar:
        st.header("SmartTrack AI")
        st.caption("Restricted Zone Intrusion Detection")
        st.divider()

        # ── Source selection ──────────────────────────────────────────
        st.subheader("Video Source")
        source_mode = st.radio(
            "Input type",
            options=["Upload video file", "Webcam"],
            index=0,
            label_visibility="collapsed",
        )

        uploaded_file = None
        webcam_index: int | None = None

        if source_mode == "Upload video file":
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=["mp4", "avi", "mov", "mkv"],
                label_visibility="collapsed",
            )
        else:
            webcam_index = st.number_input(
                "Webcam index", min_value=0, max_value=8, value=0, step=1
            )

        st.divider()

        # ── Detection settings ────────────────────────────────────────
        st.subheader("Detection")
        confidence = st.slider(
            "Confidence threshold",
            min_value=_CONFIDENCE_MIN,
            max_value=_CONFIDENCE_MAX,
            value=float(cfg["model"].get("confidence_threshold", 0.4)),
            step=_CONFIDENCE_STEP,
            help="Minimum score for a person detection to be accepted.",
        )
        # Update the live detector — takes effect on the next frame.
        if abs(detector.confidence_threshold - confidence) > 1e-6:
            detector.confidence_threshold = confidence

        st.divider()

        # ── Controls ──────────────────────────────────────────────────
        st.subheader("Controls")
        col_start, col_stop = st.columns(2)

        start_clicked = col_start.button(
            "Start",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.running,
        )
        stop_clicked = col_stop.button(
            "Stop",
            use_container_width=True,
            disabled=not st.session_state.running,
        )
        reset_clicked = st.button(
            "Reset pipeline",
            use_container_width=True,
            help="Clear tracking state and alert history without stopping.",
        )

        st.divider()

        # ── Status indicator ──────────────────────────────────────────
        if st.session_state.running:
            st.success("Running", icon="▶")
        else:
            st.info("Stopped", icon="■")

    return uploaded_file, webcam_index, start_clicked, stop_clicked, reset_clicked


# ---------------------------------------------------------------------------
# Metrics panel
# ---------------------------------------------------------------------------

def _render_metrics(result: PipelineResult | None) -> None:
    """Render the four live metric tiles.

    Args:
        result: Most recent pipeline result, or ``None`` when no frame has
            been processed yet.
    """
    fps = result.fps if result else 0.0
    people = result.people_count if result else 0
    tracks = result.active_track_count if result else 0
    intrusions = st.session_state.total_intrusions

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("FPS", f"{fps:.1f}")
    col2.metric("People detected", people)
    col3.metric("Active tracks", tracks)
    col4.metric("Total intrusions", intrusions)


# ---------------------------------------------------------------------------
# Alert panel
# ---------------------------------------------------------------------------

def _render_alerts() -> None:
    """Render the recent intrusion event table and clear button."""
    st.subheader("Intrusion Alerts")

    log: list[dict[str, Any]] = st.session_state.alert_log

    if not log:
        st.caption("No intrusion events recorded yet.")
        return

    col_table, col_clear = st.columns([5, 1])

    with col_table:
        st.dataframe(
            log[-_MAX_ALERT_ROWS:],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.TextColumn(width="small"),
                "Track ID": st.column_config.NumberColumn(width="small"),
                "Zone": st.column_config.TextColumn(width="medium"),
                "Event": st.column_config.TextColumn(width="small"),
            },
        )

    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.alert_log = []
            st.session_state.total_intrusions = 0
            st.rerun()


def _append_events_to_log(result: PipelineResult) -> None:
    """Add any new intrusion events from *result* to the session alert log.

    Args:
        result: Pipeline output for the current frame.
    """
    for event in result.intrusion_events:
        st.session_state.alert_log.append({
            "Time": event.timestamp.strftime("%H:%M:%S"),
            "Track ID": event.track_id,
            "Zone": event.zone_name,
            "Event": event.event_type.value,
        })
        if event.event_type == EventType.ENTER:
            st.session_state.total_intrusions += 1


# ---------------------------------------------------------------------------
# Video capture lifecycle
# ---------------------------------------------------------------------------

def _open_capture(source: int | str) -> cv2.VideoCapture | None:
    """Open a ``cv2.VideoCapture`` for the given source.

    Args:
        source: Webcam index (``int``) or path to video file (``str``).

    Returns:
        An opened :class:`cv2.VideoCapture`, or ``None`` when the source
        cannot be opened.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    return cap


def _release_capture() -> None:
    """Release and clear the VideoCapture stored in session state."""
    cap: cv2.VideoCapture | None = st.session_state.cap
    if cap is not None:
        cap.release()
        st.session_state.cap = None

    # Clean up any temporary file created for an uploaded video.
    tmp: str | None = st.session_state.temp_file_path
    if tmp:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass
        st.session_state.temp_file_path = None


def _resolve_source(
    uploaded_file: Any,
    webcam_index: int | None,
) -> int | str | None:
    """Determine the video source from sidebar selections.

    Saves an uploaded file to a temporary path if one was provided.

    Args:
        uploaded_file: Streamlit ``UploadedFile`` object, or ``None``.
        webcam_index: Integer webcam device index, or ``None``.

    Returns:
        Source value suitable for :func:`cv2.VideoCapture`, or ``None`` when
        no source has been selected.
    """
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()
        st.session_state.temp_file_path = tmp.name
        return tmp.name

    if webcam_index is not None:
        return int(webcam_index)

    return None


# ---------------------------------------------------------------------------
# Frame processing (one frame per Streamlit run)
# ---------------------------------------------------------------------------

def _process_next_frame(
    pipeline: InferencePipeline,
    frame_placeholder: Any,
    metrics_placeholder: Any,
) -> bool:
    """Read one frame, run the pipeline, update the UI, return success.

    Args:
        pipeline: The shared :class:`InferencePipeline` instance.
        frame_placeholder: ``st.empty()`` container for the video frame.
        metrics_placeholder: ``st.empty()`` container for live metrics.

    Returns:
        ``True`` when the frame was processed successfully; ``False`` when
        the video source is exhausted or an error occurs.
    """
    cap: cv2.VideoCapture | None = st.session_state.cap
    if cap is None or not cap.isOpened():
        return False

    ret, frame = cap.read()
    if not ret:
        return False

    result: PipelineResult = pipeline.process_frame(frame)
    st.session_state.last_result = result

    # Display annotated frame (BGR → RGB for Streamlit).
    rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

    # Append any new intrusion events to the session log.
    _append_events_to_log(result)

    # Refresh metrics tile in the placeholder.
    with metrics_placeholder.container():
        _render_metrics(result)

    return True


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the SmartTrack AI Streamlit application.

    Called on every Streamlit script execution (i.e., on every user
    interaction and on every ``st.rerun()`` triggered by the frame loop).
    """
    cfg = _load_config()

    st.set_page_config(
        page_title=cfg["app"]["title"],
        page_icon=cfg["app"]["page_icon"],
        layout=cfg["app"]["layout"],
    )

    _init_session_state()

    # ── Load pipeline (spinner shown only on first load) ──────────────
    with st.spinner("Loading YOLO model…"):
        try:
            pipeline, detector = _build_pipeline(_CONFIG_PATH)
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
            st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────
    uploaded_file, webcam_index, start_clicked, stop_clicked, reset_clicked = (
        _render_sidebar(detector)
    )

    # ── Header ────────────────────────────────────────────────────────
    st.title("SmartTrack AI")
    st.caption("Real-time restricted zone intrusion detection")
    st.divider()

    # ── Metrics placeholder (updated each frame) ──────────────────────
    metrics_placeholder = st.empty()
    with metrics_placeholder.container():
        _render_metrics(st.session_state.last_result)

    st.divider()

    # ── Video frame placeholder ───────────────────────────────────────
    frame_placeholder = st.empty()

    if not st.session_state.running:
        frame_placeholder.info(
            "Select a video source in the sidebar and press **Start**.",
            icon="ℹ",
        )

    st.divider()

    # ── Alert panel ───────────────────────────────────────────────────
    _render_alerts()

    # ────────────────────────────────────────────────────────────────────
    # Button handlers
    # These modify session state and always trigger a rerun at the end
    # of the current script execution.
    # ────────────────────────────────────────────────────────────────────

    if reset_clicked:
        pipeline.reset()
        st.session_state.alert_log = []
        st.session_state.total_intrusions = 0
        st.session_state.last_result = None
        st.toast("Pipeline reset.", icon="↺")

    if start_clicked:
        source = _resolve_source(uploaded_file, webcam_index)

        if source is None:
            st.warning("Please select a video file or webcam before starting.")
        else:
            _release_capture()  # close any previous capture
            cap = _open_capture(source)

            if cap is None:
                source_label = (
                    f"webcam {source}"
                    if isinstance(source, int)
                    else Path(str(source)).name
                )
                st.error(
                    f"Could not open source: **{source_label}**. "
                    "Check the file path or webcam connection."
                )
            else:
                pipeline.reset()
                st.session_state.cap = cap
                st.session_state.source = source
                st.session_state.running = True
                st.rerun()

    if stop_clicked:
        st.session_state.running = False
        _release_capture()
        st.rerun()

    # ── Frame loop (one frame per Streamlit run) ──────────────────────
    if st.session_state.running:
        success = _process_next_frame(pipeline, frame_placeholder, metrics_placeholder)

        if success:
            # Pace frames to ~30 FPS; yields CPU between reruns.
            time.sleep(_FRAME_DELAY_S)
            st.rerun()
        else:
            # Video exhausted or capture error — stop cleanly.
            st.session_state.running = False
            _release_capture()
            st.toast("Video ended.", icon="✓")
            st.rerun()


if __name__ == "__main__":
    main()
