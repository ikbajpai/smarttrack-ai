# SmartTrack AI

> Real-time restricted zone intrusion detection powered by YOLO, ByteTrack, OpenCV, and Streamlit.

---

## Overview

SmartTrack AI is a production-grade computer vision system that detects and tracks people (or other objects) in video streams and raises alerts when they enter user-defined restricted zones.

| Capability | Technology |
|---|---|
| Object Detection | Ultralytics YOLOv8 |
| Multi-Object Tracking | ByteTrack (via Ultralytics) |
| Video Processing | OpenCV |
| Web Interface | Streamlit |
| Configuration | YAML |
| Logging | Loguru |

---

## Project Status

| Day | Focus | Status |
|-----|-------|--------|
| Day 1 | Project initialization & architecture | ✅ Complete |
| Day 2 | Detection + Tracking pipeline | 🔜 Pending |
| Day 3 | Zone definition & intrusion logic | 🔜 Pending |
| Day 4 | Alert system & pipeline integration | 🔜 Pending |
| Day 5 | Streamlit UI & end-to-end testing | 🔜 Pending |

---

## Folder Structure

```
smarttrack-ai/
├── config/                  # Central YAML configuration
├── src/
│   ├── detection/           # YOLO detector wrapper
│   ├── tracking/            # ByteTrack tracker wrapper
│   ├── zones/               # Zone polygon management & intrusion check
│   ├── alerts/              # Alert dispatch & CSV logging
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
git clone https://github.com/your-username/smarttrack-ai.git
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

## Configuration

All runtime parameters are controlled via `config/config.yaml`:

- **model** — YOLO weights path, confidence/IoU thresholds, target device
- **tracker** — ByteTrack hyperparameters
- **zones** — Restricted zone polygon definitions
- **video** — Input source and resolution
- **alerts** — Console/file alert settings

---

## License

[MIT](LICENSE)
