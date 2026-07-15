"""
detector.py
-----------
YOLO-based person detector for SmartTrack AI.

Provides:
    Detection  — frozen dataclass representing one detected person.
    Detector   — loads a YOLO model and runs per-frame inference.

This module is intentionally isolated: it has no dependency on Streamlit,
the tracker, zone logic, or alert systems.  The pipeline layer (Day 4) wires
everything together.

Usage::

    import yaml
    import cv2
    from src.detection.detector import Detector

    with open("config/config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    detector = Detector(cfg["model"])
    detector.load()

    for annotated_frame, detections in detector.stream_frames("data/sample_videos/real/pexels_corridor.mp4"):
        cv2.imshow("SmartTrack", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# COCO class index for "person" — always 0 in the standard COCO label map.
_PERSON_CLASS_ID: int = 0

# BGR drawing colours
_BOX_COLOUR: tuple[int, int, int] = (0, 255, 0)       # green boxes
_LABEL_TEXT_COLOUR: tuple[int, int, int] = (255, 255, 255)  # white text on label
_HUD_COLOUR: tuple[int, int, int] = (0, 200, 255)     # amber HUD overlay


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """An immutable record describing one detected person in a single frame.

    Attributes:
        bbox: Bounding box in pixel coordinates as ``(x1, y1, x2, y2)``
            where *(x1, y1)* is the top-left corner and *(x2, y2)* is the
            bottom-right corner.
        confidence: Detection confidence score in the range ``[0.0, 1.0]``.
        class_id: COCO class index.  Always ``0`` (person) for this detector.
        class_name: Human-readable label string.  Always ``"person"``.

    ByteTrack compatibility:
        Call :meth:`to_bytetrack_row` to get the ``[x1, y1, x2, y2, score,
        class_id]`` array expected by ByteTrack's per-frame update matrix.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str

    def to_bytetrack_row(self) -> np.ndarray:
        """Return a float32 array ``[x1, y1, x2, y2, confidence, class_id]``.

        Used to build the ``(N, 6)`` detection matrix consumed by ByteTrack's
        ``update()`` method.

        Returns:
            A 1-D float32 NumPy array of length 6.
        """
        x1, y1, x2, y2 = self.bbox
        return np.array(
            [x1, y1, x2, y2, self.confidence, self.class_id], dtype=np.float32
        )


def detections_to_bytetrack_array(detections: list[Detection]) -> np.ndarray:
    """Stack a list of detections into a ``(N, 6)`` array for ByteTrack.

    Args:
        detections: Output of :meth:`Detector.detect`.

    Returns:
        Float32 array of shape ``(N, 6)`` — one row per detection.
        Returns an empty ``(0, 6)`` array when *detections* is empty.
    """
    if not detections:
        return np.empty((0, 6), dtype=np.float32)
    return np.vstack([d.to_bytetrack_row() for d in detections])


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class Detector:
    """YOLO-based person detector.

    Wraps an Ultralytics YOLO model, restricts inference to COCO class 0
    (person), and returns structured :class:`Detection` objects that are
    ready for downstream ByteTrack integration.

    Args:
        config: The ``model`` section of ``config/config.yaml``.  Expected
            keys::

                weights:              models/yolov8n.pt
                confidence_threshold: 0.4
                iou_threshold:        0.45
                device:               cpu   # "cpu" | "cuda" | "mps"

    Raises:
        KeyError: If the required ``weights`` key is absent from *config*.

    Example::

        import yaml
        from src.detection.detector import Detector

        with open("config/config.yaml") as fh:
            cfg = yaml.safe_load(fh)

        detector = Detector(cfg["model"])
        detector.load()
        detections = detector.detect(frame)
    """

    def __init__(self, config: dict) -> None:
        self._weights: str = config["weights"]
        self._confidence: float = float(config.get("confidence_threshold", 0.4))
        self._iou: float = float(config.get("iou_threshold", 0.45))
        self._device: str = str(config.get("device", "cpu"))
        self._model: YOLO | None = None

        # Rolling window of per-frame inference durations (seconds).
        # Used to compute a smoothed FPS reading.
        self._frame_times: deque[float] = deque(maxlen=30)

        logger.debug(
            "Detector configured | weights='{}' confidence={} iou={} device={}",
            self._weights,
            self._confidence,
            self._iou,
            self._device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load YOLO weights from disk and warm up the model.

        If the weights path does not exist locally and looks like a standard
        Ultralytics model name (e.g. ``yolov8n.pt`` with no parent directory),
        Ultralytics will download it automatically.  Paths that include a
        directory component (e.g. ``models/yolov8n.pt``) must resolve to an
        existing file.

        Raises:
            FileNotFoundError: If a directory-qualified weights path does not
                exist on disk.
            RuntimeError: If Ultralytics raises any exception during model
                initialisation.
        """
        weights_path = Path(self._weights)

        # Only enforce local-existence check when a directory is specified.
        # Bare names like "yolov8n.pt" are auto-downloaded by Ultralytics.
        if weights_path.parent != Path(".") and not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at '{weights_path.resolve()}'. "
                "Place the model file there or update 'model.weights' in "
                "config/config.yaml.  Standard weights (e.g. yolov8n.pt) "
                "can be downloaded from https://docs.ultralytics.com/models/."
            )

        logger.info("Loading YOLO model from '{}'", self._weights)
        try:
            self._model = YOLO(str(weights_path))
            self._warmup()
        except Exception as exc:
            self._model = None
            raise RuntimeError(
                f"Failed to initialise YOLO model from '{self._weights}': {exc}"
            ) from exc

        logger.info(
            "YOLO model ready | device='{}' confidence={} iou={}",
            self._device,
            self._confidence,
            self._iou,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run person detection on a single BGR frame.

        Args:
            frame: A BGR image as returned by ``cv2.VideoCapture.read()``.
                Must be a non-empty NumPy array.

        Returns:
            A list of :class:`Detection` objects, one per detected person,
            sorted by descending confidence score.  Returns an empty list when
            the frame is invalid or no persons pass the confidence threshold.

        Raises:
            RuntimeError: If :meth:`load` has not been called successfully.
        """
        self._require_loaded()

        if frame is None or frame.size == 0:
            logger.warning("detect() received an empty or None frame — skipping.")
            return []

        t0 = time.perf_counter()
        try:
            results = self._model(  # type: ignore[misc]
                frame,
                conf=self._confidence,
                iou=self._iou,
                classes=[_PERSON_CLASS_ID],
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Inference error on frame: {}", exc)
            return []

        elapsed = time.perf_counter() - t0
        self._frame_times.append(elapsed)

        detections = self._parse_results(results)
        logger.debug(
            "{} person(s) detected in {:.1f} ms",
            len(detections),
            elapsed * 1_000,
        )
        return detections

    def draw_annotations(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        *,
        show_fps: bool = True,
        show_count: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes, confidence labels, and a HUD overlay.

        The source *frame* is never mutated; a copy is returned.

        Args:
            frame: BGR frame to annotate.
            detections: Output of :meth:`detect` for this frame.
            show_fps: Render rolling-average FPS in the top-left corner.
            show_count: Render the detected person count below the FPS.

        Returns:
            A new BGR frame with all annotations applied.
        """
        annotated = frame.copy()

        for det in detections:
            self._draw_box(annotated, det)

        if show_fps:
            cv2.putText(
                annotated,
                f"FPS: {self.current_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                _HUD_COLOUR,
                2,
                cv2.LINE_AA,
            )

        if show_count:
            cv2.putText(
                annotated,
                f"Persons: {len(detections)}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                _HUD_COLOUR,
                2,
                cv2.LINE_AA,
            )

        return annotated

    def stream_frames(
        self,
        source: int | str,
    ) -> Generator[tuple[np.ndarray, list[Detection]], None, None]:
        """Yield ``(annotated_frame, detections)`` for every frame in *source*.

        Handles both webcam indices (``int``) and file paths (``str``).  The
        underlying :class:`cv2.VideoCapture` is always released — even when
        the caller closes the generator early via ``generator.close()``.

        Args:
            source: ``0`` for the default webcam, or an absolute/relative path
                to a video file (MP4, AVI, MOV, …).

        Yields:
            A ``(annotated_frame, detections)`` tuple for each readable frame.
            *annotated_frame* is the BGR frame with boxes and HUD drawn.
            *detections* is the raw list of :class:`Detection` objects for
            that frame, useful for passing to the tracker.

        Raises:
            RuntimeError: If :meth:`load` has not been called.
            ValueError: If *source* cannot be opened by OpenCV.

        Example::

            for annotated, dets in detector.stream_frames(0):  # webcam
                cv2.imshow("SmartTrack", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
        """
        self._require_loaded()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(
                f"Cannot open video source {source!r}. "
                "Check that the file exists or the webcam index is valid."
            )

        source_label = (
            f"webcam:{source}" if isinstance(source, int) else Path(str(source)).name
        )
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        logger.info(
            "Opening source '{}' | native_fps={:.1f} total_frames={}",
            source_label,
            native_fps,
            total if total > 0 else "∞",
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of stream for '{}'.", source_label)
                    break
                detections = self.detect(frame)
                annotated = self.draw_annotations(frame, detections)
                yield annotated, detections
        finally:
            cap.release()
            logger.debug("VideoCapture released for '{}'.", source_label)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """``True`` after :meth:`load` has completed successfully."""
        return self._model is not None

    @property
    def current_fps(self) -> float:
        """Smoothed inference throughput over the last 30 frames.

        Returns ``0.0`` before any frames have been processed.
        """
        if not self._frame_times:
            return 0.0
        avg = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg if avg > 0.0 else 0.0

    @property
    def confidence_threshold(self) -> float:
        """Current confidence threshold (read/write)."""
        return self._confidence

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        """Set a new confidence threshold at runtime.

        Args:
            value: New threshold, must be in the open interval ``(0.0, 1.0)``.

        Raises:
            ValueError: If *value* is outside ``(0.0, 1.0)``.
        """
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0.0, 1.0), got {value!r}."
            )
        logger.debug(
            "Confidence threshold changed: {} → {}", self._confidence, value
        )
        self._confidence = value

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        """Raise :exc:`RuntimeError` if the model has not been loaded yet."""
        if self._model is None:
            raise RuntimeError(
                "Detector model is not loaded. Call Detector.load() before "
                "running inference."
            )

    def _warmup(self) -> None:
        """Run a single dummy inference to initialise CUDA/MPS contexts.

        This avoids an artificially slow first frame during live video.
        The dummy frame is a black 640×640 image.
        """
        logger.debug("Warming up model on device='{}'.", self._device)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy, device=self._device, verbose=False)  # type: ignore[misc]
        self._frame_times.clear()  # don't count warm-up in FPS tracking
        logger.debug("Warm-up complete.")

    def _parse_results(self, results: list) -> list[Detection]:
        """Convert raw Ultralytics result objects to :class:`Detection` list.

        Args:
            results: The list returned by invoking a :class:`ultralytics.YOLO`
                instance on a frame.

        Returns:
            List of :class:`Detection`, sorted by descending confidence.
        """
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()     # (N, 4) float32
            confs = boxes.conf.cpu().numpy()    # (N,)   float32
            cls_ids = boxes.cls.cpu().numpy()   # (N,)   float32

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name="person",
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _draw_box(self, frame: np.ndarray, det: Detection) -> None:
        """Draw a single bounding box and label onto *frame* in-place.

        Args:
            frame: BGR frame to annotate (mutated directly).
            det: Detection to render.
        """
        x1, y1, x2, y2 = det.bbox
        label = f"person {det.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOUR, 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (x1 + text_w, label_top + text_h + baseline + 4),
            _BOX_COLOUR,
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
