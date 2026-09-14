"""YOLO detector implementation (Ultralytics).

Degrades gracefully (coding rule #9, "handle hardware absence gracefully" —
applied here to *model* absence too) when ultralytics/torch aren't
installed or a GPU isn't available: it logs once and returns an empty
detection list rather than crashing the pipeline, so simulation mode still
runs end-to-end on a machine with no CUDA and no internet to fetch weights.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from common.types import BBox, Detection2D
from perception.detector.base_detector import BaseObjectDetector

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO

    _HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover
    _HAS_ULTRALYTICS = False
    logger.warning("ultralytics not installed; YOLODetector will run in no-op mode.")


class YOLODetector(BaseObjectDetector):
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cpu",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        max_detections: int = 50,
        image_size: int = 640,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.max_detections = max_detections
        self.image_size = image_size
        self._model = None
        self._names: dict[int, str] = {}
        self._degraded = False
        self._last_inference_ms: float = 0.0

    def initialize(self) -> None:
        if not _HAS_ULTRALYTICS:
            self._degraded = True
            return

        self.device = self._resolve_device(self.device)

        try:
            self._model = YOLO(self.model_path)
            self._names = self._model.names
            logger.info("YOLODetector initialized: model=%s device=%s imgsz=%d",
                        self.model_path, self.device, self.image_size)
        except Exception as exc:  # noqa: BLE001 - any load failure -> degrade, don't crash
            logger.error("Failed to load YOLO model '%s': %s. Detector running in "
                        "no-op mode.", self.model_path, exc)
            self._degraded = True
            self._model = None

    def infer(self, frame: np.ndarray) -> list[Detection2D]:
        if self._degraded or self._model is None:
            return []

        started = time.perf_counter()
        try:
            results = self._model.predict(
                source=frame,
                device=self.device,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                max_det=self.max_detections,
                imgsz=self.image_size,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO inference failed on this frame: %s", exc)
            self._last_inference_ms = (time.perf_counter() - started) * 1000.0
            return []
        finally:
            self._last_inference_ms = (time.perf_counter() - started) * 1000.0

        detections: list[Detection2D] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = self._names.get(cls_id, str(cls_id))
            detections.append(
                Detection2D(
                    bbox=BBox(*xyxy),
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                )
            )

        return detections

    def shutdown(self) -> None:
        self._model = None

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        """Fall back to CPU if a CUDA device was requested but none is
        available, instead of letting every single inference call fail —
        this is the "handle hardware absence gracefully" rule applied to
        compute, not just sensors. A Jetson/GPU workstation with CUDA
        present keeps using the configured device unchanged.
        """
        if not requested_device.startswith("cuda"):
            return requested_device

        try:
            import torch

            if torch.cuda.is_available():
                return requested_device
        except ImportError:
            pass

        logger.warning("Requested device '%s' but no CUDA GPU is available; "
                       "falling back to CPU inference.", requested_device)
        return "cpu"

    @property
    def is_degraded(self) -> bool:
        """True when running in no-op mode (model unavailable)."""
        return self._degraded

    @property
    def last_inference_ms(self) -> float:
        """Wall time of the most recent infer() call, for pipeline timing."""
        return self._last_inference_ms

    def get_model_info(self) -> dict:
        """Provenance for diagnostics: what model is actually running, where."""
        return {
            "backend": "ultralytics" if _HAS_ULTRALYTICS else "unavailable",
            "model_path": self.model_path,
            "device": self.device,
            "image_size": self.image_size,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "degraded": self._degraded,
            "num_classes": len(self._names) if self._names else 0,
        }


def build_detector_from_config(cfg: dict) -> BaseObjectDetector:
    p_cfg = cfg["perception"]["detector"]
    return YOLODetector(
        model_path=p_cfg.get("model_path", "yolov8n.pt"),
        device=p_cfg.get("device", "cpu"),
        confidence_threshold=p_cfg.get("confidence_threshold", 0.35),
        iou_threshold=p_cfg.get("iou_threshold", 0.45),
        classes=p_cfg.get("classes"),
        max_detections=p_cfg.get("max_detections", 50),
        image_size=p_cfg.get("image_size", 640),
    )
