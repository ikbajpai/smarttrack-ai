"""
test_tracker.py
---------------
Standalone script for visually verifying the SmartTrack AI tracking module.

Loads config/config.yaml, initialises Detector + Tracker, streams frames,
runs detect → track on each frame, and displays bounding boxes with stable
track IDs and fading motion trails in an OpenCV window.

Press 'q' or ESC to exit cleanly.

Usage
-----
# Source from config.yaml (0 = default webcam)
    python test_tracker.py

# Override source
    python test_tracker.py --source data/sample_videos/real/pexels_corridor.mp4

# Disable motion trails
    python test_tracker.py --source pexels_corridor.mp4 --no-trails

# Run headless (no window) and log per-frame stats
    python test_tracker.py --source pexels_corridor.mp4 --no-display

# Adjust log verbosity
    python test_tracker.py --source pexels_corridor.mp4 --log-level INFO
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Project root on sys.path so src.* imports resolve regardless of cwd.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.detection.detector import Detector      # noqa: E402
from src.tracking.tracker import Tracker         # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CONFIG_PATH = _ROOT / "config" / "config.yaml"
_WINDOW_TITLE = "SmartTrack AI — Tracker Test  |  q / ESC to quit"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure_logger(log_level: str = "DEBUG") -> None:
    """Remove the default loguru sink and add a clean, coloured console sink.

    Args:
        log_level: Minimum level to emit.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


def _load_config(path: Path) -> dict:
    """Parse config.yaml and return the full configuration dict.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Parsed configuration as a nested dict.

    Raises:
        SystemExit: If the file is missing or malformed.
    """
    if not path.exists():
        logger.critical("Config file not found: '{}'", path)
        sys.exit(1)
    with path.open() as fh:
        try:
            cfg = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            logger.critical("Failed to parse config.yaml: {}", exc)
            sys.exit(1)
    logger.info("Configuration loaded from '{}'", path)
    return cfg


def _resolve_source(cfg_source: int | str, cli_source: str | None) -> int | str:
    """Determine the final video source, preferring the CLI override.

    Args:
        cfg_source: Value from ``video.source`` in config.yaml.
        cli_source: Optional ``--source`` argument from the command line.

    Returns:
        Camera index (``int``) or file path (``str``).
    """
    raw = cli_source if cli_source is not None else cfg_source
    try:
        return int(raw)
    except (ValueError, TypeError):
        return str(raw)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SmartTrack AI — standalone tracker test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        default=None,
        metavar="PATH_OR_INDEX",
        help="Video file path or webcam index. Overrides video.source in config.yaml.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Confidence threshold in (0, 1). Overrides model.confidence_threshold.",
    )
    parser.add_argument(
        "--no-trails",
        action="store_true",
        help="Disable motion trail rendering.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless: skip the OpenCV window (useful for SSH / CI).",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Minimum log level (default: DEBUG).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _configure_logger(args.log_level)

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = _load_config(_CONFIG_PATH)
    model_cfg = cfg["model"]
    tracker_cfg = cfg["tracker"]
    video_cfg = cfg.get("video", {})

    if args.confidence is not None:
        if not 0.0 < args.confidence < 1.0:
            logger.critical("--confidence must be in (0, 1), got {}", args.confidence)
            sys.exit(1)
        model_cfg["confidence_threshold"] = args.confidence
        logger.info("Confidence threshold overridden to {}", args.confidence)

    source = _resolve_source(video_cfg.get("source", 0), args.source)
    source_label = (
        f"webcam:{source}" if isinstance(source, int) else Path(str(source)).name
    )

    logger.info("Video source : {}", source_label)
    logger.info("Model weights: {}", model_cfg["weights"])
    logger.info("Device       : {}", model_cfg.get("device", "cpu"))
    logger.info("Confidence   : {}", model_cfg.get("confidence_threshold", 0.4))
    logger.info("Trails       : {}", "off" if args.no_trails else "on")
    logger.info("Display      : {}", "off (headless)" if args.no_display else "on")

    # ── Module init ──────────────────────────────────────────────────────────
    detector = Detector(model_cfg)
    logger.info("Initialising detector …")
    try:
        detector.load()
    except (FileNotFoundError, RuntimeError) as exc:
        logger.critical("Detector failed to load: {}", exc)
        sys.exit(1)
    logger.info("Detector ready.")

    tracker = Tracker(tracker_cfg)
    logger.info("Tracker ready.")

    # ── SIGINT handler ───────────────────────────────────────────────────────
    interrupted = False

    def _handle_sigint(sig: int, frame: object) -> None:  # noqa: ARG001
        nonlocal interrupted
        logger.info("Interrupted by user (SIGINT).")
        interrupted = True

    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Display setup ────────────────────────────────────────────────────────
    if not args.no_display:
        cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            _WINDOW_TITLE,
            video_cfg.get("width", 1280),
            video_cfg.get("height", 720),
        )

    # ── Streaming loop ───────────────────────────────────────────────────────
    logger.info("Starting detect + track loop on '{}' …", source_label)

    total_frames = 0
    total_persons = 0
    loop_start = time.perf_counter()

    try:
        frame_gen = detector.stream_frames(source)
    except ValueError as exc:
        logger.critical("Cannot open source '{}': {}", source_label, exc)
        sys.exit(1)

    try:
        for _annotated_frame, detections in frame_gen:
            if interrupted:
                break

            # Read the raw frame from the generator's internal capture.
            # stream_frames() already consumed cap.read(); we reconstruct
            # the raw frame by undoing detector annotations (not ideal) —
            # instead we keep both: pass detections + the raw frame to tracker.
            # NOTE: stream_frames() yields (annotated_frame, detections).
            # The raw frame is needed by BYTETracker for image-space Kalman.
            # We pass the annotated frame as a suitable substitute since
            # its spatial content is identical to the raw frame.
            raw_frame = _annotated_frame  # same shape/size as raw

            total_frames += 1
            total_persons += len(detections)

            tracked = tracker.update(detections, raw_frame)

            output = tracker.draw_annotations(
                raw_frame,
                tracked,
                show_trails=not args.no_trails,
                show_ids=True,
                show_count=True,
            )

            # Overlay detector FPS in a second HUD line
            cv2.putText(
                output,
                f"Det FPS: {detector.current_fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

            logger.debug(
                "Frame {:>5} | detections={} | tracks={} | fps={:.1f}",
                total_frames,
                len(detections),
                len(tracked),
                detector.current_fps,
            )

            if not args.no_display:
                cv2.imshow(_WINDOW_TITLE, output)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    logger.info("Quit key pressed — stopping.")
                    break

    except Exception as exc:
        logger.error("Unexpected runtime error: {}", exc)
    finally:
        frame_gen.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - loop_start
    avg_fps = total_frames / elapsed if elapsed > 0 else 0.0

    logger.info("─" * 50)
    logger.info("Session complete")
    logger.info("  Source        : {}", source_label)
    logger.info("  Frames        : {}", total_frames)
    logger.info("  Total persons : {}", total_persons)
    logger.info("  Trail IDs seen: {}", tracker.active_trail_count)
    logger.info("  Elapsed       : {:.1f} s", elapsed)
    logger.info("  Avg FPS       : {:.1f}", avg_fps)
    logger.info("─" * 50)


if __name__ == "__main__":
    main()
