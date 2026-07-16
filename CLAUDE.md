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

**`src/zones/zone_manager.py`** — Zone polygon management stub. Zones are defined as polygons in `config/config.yaml` and as JSON files in `data/zone_configs/`.

**`src/alerts/alert_manager.py`** — Alert dispatch stub. Alerts are logged to CSV in `outputs/intrusion_logs/`.

**`src/pipeline/inference_pipeline.py`** — Central orchestration stub that will tie all modules together.

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
- **Day 3** (pending): Zone definition & intrusion logic
- **Day 4** (pending): Alert system & pipeline integration
- **Day 5** (pending): Streamlit UI & end-to-end testing

Stub modules have fully documented method signatures and expected input/output formats — implement those contracts when filling in the stubs.

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

### 🔄 Day 2
- ByteTrack integration
- Stable tracking IDs
- Motion trails
- Tracker testing

### Day 3
- Polygon zone drawing
- Intrusion engine
- Zone testing

### Day 4
- Pipeline integration
- Streamlit dashboard
- Webcam support
- Video upload

### Day 5
- CSV logging
- Documentation
- Screenshots
- Demo video
- Final polish

