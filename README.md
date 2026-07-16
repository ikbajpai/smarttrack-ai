# SmartTrack AI

> Real-time restricted zone intrusion detection powered by YOLO, ByteTrack, OpenCV, and Streamlit.

---

## Overview

SmartTrack AI is a production-grade computer vision system that detects and tracks people (or other objects) in video streams and raises alerts when they enter user-defined restricted zones.

| Capability | Technology |
|---|---|
| Object Detection | Ultralytics YOLOv8 |
| Multi-Object Tracking | ByteTrack (via Ultralytics) |
| Zone Geometry | Shapely |
| Video Processing | OpenCV |
| Web Interface | Streamlit |
| Configuration | YAML |
| Logging | Loguru |

---

## Project Status

| Day | Focus | Status |
|-----|-------|--------|
| Day 1 | Project architecture, YOLO detector, standalone test script | ✅ Complete |
| Day 2 | ByteTrack integration, stable track IDs, motion trails | ✅ Complete |
| Day 3 | Zone polygon management, intrusion detection, 53 zone tests | ✅ Complete |
| Day 4 | Alert system, inference pipeline, 81 new tests (168 total) | ✅ Complete |
| Day 5 | Streamlit UI — live feed, metrics, alert table, source switching | ✅ Complete |

---

## UI Features

The Streamlit dashboard (`app/streamlit_app.py`) provides:

| Panel | Details |
|---|---|
| **Sidebar — Source** | Upload MP4/AVI/MOV/MKV or select a webcam index |
| **Sidebar — Detection** | Live confidence threshold slider (updates mid-stream) |
| **Sidebar — Controls** | Start, Stop, and Reset pipeline buttons |
| **Video feed** | Annotated frames with bounding boxes, motion trails, and zone overlays |
| **Metrics** | FPS · People detected · Active tracks · Total intrusions |
| **Alert table** | Recent ENTER/EXIT events with timestamp, track ID, and zone name |

---

## Folder Structure

```
smarttrack-ai/
├── config/                  # Central YAML configuration
├── src/
│   ├── detection/           # YOLO detector wrapper
│   ├── tracking/            # ByteTrack tracker wrapper
│   ├── zones/               # Zone polygon management & intrusion check
│   ├── alerts/              # Alert dispatch with pluggable handler backends
│   ├── pipeline/            # Inference orchestration
│   └── utils/               # Video helpers & logger
├── app/                     # Streamlit application
├── models/                  # YOLO weight files (not tracked in git)
├── data/
│   ├── sample_videos/       # Test footage (not tracked in git)
│   └── zone_configs/        # Zone polygon JSON definitions
├── outputs/
│   ├── videos/              # Processed output videos
│   ├── screenshots/         # Captured frames
│   └── intrusion_logs/      # Intrusion event CSV logs
├── assets/screenshots/      # UI screenshots for documentation
├── logs/                    # Runtime logs (not tracked in git)
└── tests/                   # Pytest test suite
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ikbajpai/smarttrack-ai.git
cd smarttrack-ai
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---

## Testing Modules

Standalone scripts let you verify each module independently against real video:

```bash
# Test detector only
python test_detector.py --source data/sample_videos/real/pexels_corridor.mp4

# Test detector + tracker (with motion trails)
python test_tracker.py --source data/sample_videos/real/pexels_shopping_mall.mp4

# Headless / CI mode
python test_tracker.py --source data/sample_videos/real/pexels_corridor.mp4 --no-display --log-level INFO
```

Run the full unit test suite (168 tests across detector, tracker, zone manager, alert manager, and pipeline):

```bash
pytest tests/
pytest tests/ --cov=src --cov-report=html
```

---

## Configuration

All runtime parameters are controlled via `config/config.yaml`:

- **model** — YOLO weights path, confidence/IoU thresholds, target device (`cpu` / `cuda` / `mps`)
- **tracker** — ByteTrack hyperparameters (`track_high_thresh`, `match_thresh`, `fuse_score`, trail length/colour, etc.)
- **zones** — Restricted zone polygon definitions with per-zone alert cooldown (also loadable from `data/zone_configs/*.json`)
- **video** — Input source (`0` = webcam or file path) and resolution
- **alerts** — Console alert toggle, deduplication cooldown (`cooldown_seconds`), and in-memory history cap (`max_history`)

---

## License

[MIT](LICENSE)
