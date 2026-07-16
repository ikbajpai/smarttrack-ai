# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmartTrack AI is a real-time restricted zone intrusion detection system. It uses YOLO v8 (Ultralytics) for person detection, ByteTrack for multi-object tracking, OpenCV for video processing, and Streamlit for the web UI. The pipeline detects people in video streams and triggers alerts when they enter user-defined restricted polygon zones.

## Commands

### Setup
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app/streamlit_app.py
```

### Test the Detector (standalone script)
```bash
python test_detector.py
python test_detector.py --source data/sample_videos/real/pexels_corridor.mp4
python test_detector.py --source pexels_corridor.mp4 --confidence 0.5 --no-display --log-level INFO
```

### Test the Tracker (standalone script)
```bash
python test_tracker.py
python test_tracker.py --source data/sample_videos/real/pexels_corridor.mp4
python test_tracker.py --source pexels_corridor.mp4 --no-trails --no-display --log-level INFO
```

### Run Tests
```bash
pytest tests/
pytest tests/ --cov=src --cov-report=html
```

## Architecture

The pipeline flows sequentially through these modules in `src/`:

```
Detector → Tracker → ZoneManager → AlertManager
                          ↑
                   InferencePipeline (orchestrates all)
                          ↓
                   Streamlit App (UI)
```

**`src/detection/detector.py`** — Only fully implemented module. `Detector` wraps the Ultralytics YOLO model, filters results to COCO class 0 (persons only), and returns immutable `Detection` dataclass instances `(bbox, confidence, class_id, class_name)`. `stream_frames()` is a generator for video streaming. `detections_to_bytetrack_array()` converts detections to the format expected by the tracker.

**`src/tracking/tracker.py`** — Fully implemented. `Tracker` wraps `BYTETracker` (via Ultralytics, no separate package). Accepts `list[Detection]` from the detector, converts bounding boxes to center-format xywh via `_BoxProxy`, and returns `list[TrackedObject]` (frozen dataclass with `track_id`, `bbox`, `centroid`, `confidence`, `class_name`). Internally maintains a `dict[track_id → deque[centroid]]` for fading motion trail rendering. `update()` parses the `(N, 8)` numpy output from `BYTETracker._format_output()`. `reset()` clears trails and re-initialises BYTETracker for new video sources.

**`src/zones/zone_manager.py`** — Fully implemented. `ZoneManager` loads named polygon zones from `config/config.yaml` and/or JSON files in `data/zone_configs/`. Uses Shapely for point-in-polygon tests against each `TrackedObject.foot_point` (bottom-centre of bbox). Returns `list[IntrusionEvent]` (frozen dataclass with `timestamp`, `track_id`, `zone_name`, `event_type`). Maintains `_active_intrusions: set[tuple[int, str]]` to deduplicate ENTER events and emit EXIT when a person leaves or their track is dropped. `draw_zones()` renders semi-transparent polygon overlays. `load_zones_from_file()` / `save_zones_to_file()` handle JSON round-trips with graceful error handling.

**`src/alerts/alert_manager.py`** — Fully implemented. `AlertManager` receives `list[IntrusionEvent]`, deduplicates by `(track_id, zone_name, event_type)` within a configurable cooldown window, and dispatches to registered `AlertHandler` backends. `AlertHandler` is an ABC — new backends (CSV, Telegram, Webhook) are added by subclassing without changing the public interface. `ConsoleAlertHandler` (built-in) logs ENTER at WARNING and EXIT at INFO. Maintains a bounded `deque[IntrusionEvent]` history; `get_recent(n)` returns newest-first.

**`src/pipeline/inference_pipeline.py`** — Fully implemented. `InferencePipeline` receives pre-built module instances (dependency injection) and calls them in sequence: Detect → Track → Zone check → Alert → Annotate. Returns a `PipelineResult` frozen dataclass (`annotated_frame`, `tracked_objects`, `intrusion_events`, `people_count`, `active_track_count`, `fps`, `processing_time_ms`). Each stage is individually guarded with try/except so one failing module never crashes the pipeline. `reset()` delegates to tracker, zone manager, and alert manager for clean source switching.

**`app/streamlit_app.py`** — Fully implemented. Thin presentation layer that interacts exclusively with `InferencePipeline`. Uses `@st.cache_resource` to load YOLO weights once per server process. Processes one frame per Streamlit script execution and calls `st.rerun()` for the next frame, keeping the Stop button and sidebar controls responsive. Sidebar: video upload, webcam selection, confidence slider, Start/Stop/Reset. Main area: live annotated video feed, four metric tiles (FPS, people, tracks, intrusions), recent alert table with clear button.

**`src/utils/logger.py`** — Loguru configuration with colored output.

## Configuration

All runtime parameters live in `config/config.yaml`:
- `model`: weights path, confidence threshold (0.4), IoU (0.45), device (`cpu`/`cuda`/`mps`)
- `tracker`: ByteTrack hyperparameters
- `zones`: polygon definitions with intrusion cooldown timers
- `video`: input source (`0` = webcam), resolution, FPS
- `alerts`: console/file logging, CSV output path

## Sample Data

Test surveillance videos are in `data/sample_videos/real/` (8 real CCTV clips: `pexels_corridor.mp4`, `pexels_warehouse.mp4`, etc.) and `data/sample_videos/` (synthetic: `warehouse.mp4`, `parking.mp4`, etc.). YOLO nano weights (`models/yolov8n.pt`) are committed to the repo.

## Implementation Status

- **Day 1** (complete): Project structure, Detector module, standalone test script
- **Day 2** (complete): Tracker module, motion trails, 34 unit tests, standalone test script
- **Day 3** (complete): ZoneManager — polygon loading, Shapely intrusion engine, 53 unit tests
- **Day 4** (complete): AlertManager, InferencePipeline, PipelineResult — 81 new tests (168 total)
- **Day 5** (complete): Streamlit UI — live video feed, metrics, alert table, source switching

## Engineering Principles

- Never rewrite a working module unless explicitly requested.
- Implement one module at a time.
- Maintain loose coupling between modules.
- Detector must never depend on Tracker.
- Tracker must never depend on ZoneManager.
- ZoneManager must never depend on Streamlit.
- Prefer composition over inheritance.
- Keep public interfaces backward compatible.
- Use dependency injection where appropriate.
- Use logging instead of print statements.
- Every public function must include type hints and Google-style docstrings.
- Read all runtime parameters from config/config.yaml.
- Follow PEP 8 and production-quality Python practices.

## Git Workflow

For every completed feature:

1. Test the implementation.
2. Review the generated code.
3. Commit a single logical feature.
4. Push to GitHub.

Commit messages should describe one completed feature only.

## Development Roadmap

### ✅ Day 1
- Project architecture
- Detector
- Test application
- Sample videos

### ✅ Day 2
- ByteTrack integration
- Stable tracking IDs
- Motion trails
- Tracker testing

### ✅ Day 3
- Polygon zone drawing (Shapely)
- Intrusion engine (ENTER/EXIT events)
- Zone testing (53 tests)

### ✅ Day 4
- AlertManager with pluggable handler backends
- InferencePipeline orchestration
- PipelineResult dataclass
- 81 new tests (168 total)

### ✅ Day 5
- Streamlit UI (live feed, metrics, alerts)
- Video upload + webcam selection
- Confidence slider with live update
- Start / Stop / Reset controls

