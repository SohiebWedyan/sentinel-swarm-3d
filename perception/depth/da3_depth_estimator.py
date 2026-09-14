"""Depth Anything (DA3) monocular depth estimator.

Backends, selected by perception.depth.backend:

    "transformers" / "auto"  -- real Depth Anything checkpoint via
                                HuggingFace transformers. Model is loaded
                                once at initialize(); inference runs under
                                torch.inference_mode() on the configured
                                device (CUDA on Jetson, CPU fallback).
    "mock"                   -- synthetic generator so the pipeline is
                                runnable with no model and no GPU.

WHAT THE NUMBERS MEAN: this is MONOCULAR depth. The output is RELATIVE —
internally consistent within a frame, unitless, and up to an unknown scale
that can change frame to frame. get_depth_kind() reports
DepthKind.RELATIVE, and nothing here converts to or claims meters. Metric
3D localization needs either a metric source (stereo / RGB-D / a
metric-finetuned model) or a scale solved against a known reference; until
then, treating these values as distances produces confident wrong answers.

Two execution paths, for two different callers:

    infer() + get_depth()  -- asynchronous. A background thread runs the
                              model at inference_frequency_hz and the caller
                              never blocks. Right for a control loop; the
                              map returned may be from an older frame.
    infer_sync()           -- synchronous. Returns the depth map for THIS
                              frame. Used by runtime/perception_process.py,
                              which must pair depth with the detections from
                              the same frame.

TensorRT note: on Jetson the fastest path is a prebuilt .engine rather than
the PyTorch path here (prior measurement: ~109 ms, ~6 FPS at 504x280). That
belongs behind this same interface as a sibling backend class; it is not
implemented here because it cannot be built or verified without the device.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

from common.types import DepthKind
from perception.depth.base_depth_estimator import BaseDepthEstimator

logger = logging.getLogger(__name__)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


class _MockDA3Backend:
    """Deterministic, fast, GPU-free stand-in for the real DA3 model.

    Produces a depth map that at least *looks* like monocular depth (closer
    at the bottom of the frame, objects the synthetic camera drew showing up
    as distinct depth blobs via simple intensity heuristics) so downstream
    median-depth-in-bbox logic gets non-degenerate values during development
    and CI, without requiring any model weights.

    The values are invented. They are in no sense measurements of a real
    scene, and like the real model's output they are reported as
    DepthKind.RELATIVE — the numeric range merely happens to be plausible.
    """

    def __init__(self, resolution: tuple[int, int]) -> None:
        self.resolution = resolution  # (w, h)

    def infer(self, frame: np.ndarray) -> np.ndarray:
        h, w = self.resolution[1], self.resolution[0]
        if _HAS_CV2:
            small = cv2.resize(frame, (w, h))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            # Nearest-neighbor downsample without cv2, then fake luminance.
            src_h, src_w = frame.shape[:2]
            ys = (np.linspace(0, src_h - 1, h)).astype(int)
            xs = (np.linspace(0, src_w - 1, w)).astype(int)
            small = frame[np.ix_(ys, xs)]
            gray = small.mean(axis=2).astype(np.float32)

        # Row gradient: bottom of frame = near (small depth), top = far.
        row_gradient = np.linspace(8.0, 1.0, h, dtype=np.float32).reshape(h, 1)
        base_depth = np.tile(row_gradient, (1, w))

        # Brighter synthetic "objects" pull depth closer, adding a bit of
        # per-pixel texture so a bbox median isn't perfectly uniform.
        normalized_luma = gray / 255.0
        depth = base_depth - (normalized_luma * 2.5)
        depth = np.clip(depth, 0.2, 12.0)
        return depth.astype(np.float32)


class _TransformersDA3Backend:
    """Real monocular depth via the HuggingFace `transformers` Depth Anything
    checkpoints (V2 small/base/large, or a local export directory).

    Contract notes that matter:
      * the model is loaded ONCE, in load() — never per frame
      * inference runs under torch.inference_mode() (no autograd graph)
      * the output is RELATIVE inverse depth: internally consistent within a
        frame, unitless, and up to an unknown per-frame scale. It is
        returned as-is and labelled DepthKind.RELATIVE. It is NOT converted
        to, or described as, meters anywhere.

    TensorRT note: on Jetson the fastest path is a prebuilt .engine rather
    than this PyTorch path. That belongs behind the same interface as a
    sibling backend class; it is not implemented here because it cannot be
    built or tested without the device.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        precision: str,
        resolution: tuple[int, int],
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.precision = precision
        self.resolution = resolution
        self._model = None
        self._processor = None
        self._torch_dtype = None

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._processor = AutoImageProcessor.from_pretrained(self.model_path)
        model = AutoModelForDepthEstimation.from_pretrained(self.model_path)

        use_half = self.precision == "fp16" and str(self.device).startswith("cuda")
        self._torch_dtype = torch.float16 if use_half else torch.float32
        if use_half:
            model = model.half()

        model = model.to(self.device)
        model.eval()
        self._model = model
        logger.info("DA3 transformers backend loaded: model=%s device=%s dtype=%s",
                    self.model_path, self.device, self._torch_dtype)

    def infer(self, frame: np.ndarray) -> np.ndarray:
        w, h = self.resolution

        # BGR (OpenCV convention used throughout this project) -> RGB, and
        # downscale to the configured inference resolution before the model
        # rather than after, so we pay for the pixels we actually use.
        if _HAS_CV2:
            resized = cv2.resize(frame, (w, h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:  # pragma: no cover - opencv is a hard dep for the real backend
            rgb = frame[..., ::-1]

        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self._torch_dtype == torch.float16:
            inputs = {k: (v.half() if v.is_floating_point() else v) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)
            predicted = outputs.predicted_depth

        depth = predicted.squeeze().to(torch.float32).cpu().numpy()

        if depth.shape != (h, w) and _HAS_CV2:
            depth = cv2.resize(depth, (w, h))
        return depth.astype(np.float32)

    def describe(self) -> dict:
        return {
            "backend": "transformers",
            "model": self.model_path,
            "device": self.device,
            "precision": self.precision,
        }


class DA3DepthEstimator(BaseDepthEstimator):
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        precision: str = "fp32",
        inference_resolution: tuple[int, int] = (504, 280),
        inference_frequency_hz: float = 6.0,
        max_queue_size: int = 2,
        backend: str = "auto",
        model_path: str = "",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.precision = precision
        self.inference_resolution = tuple(inference_resolution)
        self.inference_frequency_hz = max(inference_frequency_hz, 0.1)
        self.max_queue_size = max_queue_size
        self.backend_name = backend
        self.model_path = model_path

        self._backend = None
        self._using_mock = True
        self._last_inference_ms: float = 0.0
        self._load_error: Optional[str] = None

        self._input_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=self.max_queue_size)
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_confidence: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        self._backend = self._load_backend()
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="DA3Worker")
        self._worker.start()
        logger.info(
            "DA3DepthEstimator initialized (mock=%s, model=%s, device=%s, "
            "resolution=%s, depth_kind=%s, target_hz=%.1f)",
            self._using_mock, self.model_path or "mock-da3", self.device,
            self.inference_resolution, self.get_depth_kind().value,
            self.inference_frequency_hz,
        )
        if self._using_mock:
            logger.warning(
                "DA3 is running the MOCK depth generator — its output is synthetic and "
                "carries no information about the real scene."
            )

    def _load_backend(self):
        """Select and load the depth backend.

        Selection is explicit rather than magical:
          backend: "mock"          -> always the mock generator
          backend: "transformers"  -> require the real model; mock only if it fails
          backend: "auto"          -> real model IF a model_path is configured
                                      AND transformers/torch are importable,
                                      otherwise the mock

        "auto" deliberately requires an explicit model_path: silently
        reaching out to download a checkpoint the operator never named would
        be a surprising side effect of merely starting the process, and it
        would make test runs depend on the network.
        """
        if self.backend_name == "mock":
            self._using_mock = True
            return _MockDA3Backend(self.inference_resolution)

        wants_real = self.backend_name == "transformers" or (
            self.backend_name == "auto" and bool(self.model_path)
        )

        if not wants_real:
            self._using_mock = True
            logger.info("DA3: no model_path configured; using the mock depth generator. "
                        "Set perception.depth.model_path to run a real model.")
            return _MockDA3Backend(self.inference_resolution)

        if not self.model_path:
            self._load_error = "backend='transformers' requires perception.depth.model_path"
            logger.error("DA3: %s; falling back to the mock depth generator.", self._load_error)
            self._using_mock = True
            return _MockDA3Backend(self.inference_resolution)

        if not _HAS_TORCH:
            self._load_error = "torch is not installed"
            logger.error("DA3: %s; falling back to the mock depth generator.", self._load_error)
            self._using_mock = True
            return _MockDA3Backend(self.inference_resolution)

        try:
            device = self._resolve_device(self.device)
            backend = _TransformersDA3Backend(
                model_path=self.model_path,
                device=device,
                precision=self.precision,
                resolution=self.inference_resolution,
            )
            backend.load()
            self.device = device
            self._using_mock = False
            self._load_error = None
            return backend
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "DA3: failed to load real backend '%s' (%s); falling back to the mock depth "
                "generator. Depth output is SYNTHETIC and must not be used for anything "
                "that matters.", self.model_path, self._load_error,
            )
            self._using_mock = True
            return _MockDA3Backend(self.inference_resolution)

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        """Same graceful-degradation rule the YOLO detector uses: a CUDA
        device that isn't there becomes CPU once, at load time, rather than
        failing every single inference call.
        """
        if not str(requested_device).startswith("cuda"):
            return requested_device
        try:
            if torch.cuda.is_available():
                return requested_device
        except Exception:  # noqa: BLE001
            pass
        logger.warning("DA3: requested device '%s' but no CUDA GPU is available; "
                       "falling back to CPU inference.", requested_device)
        return "cpu"

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self._backend = None

    # -- async inference -----------------------------------------------------

    def infer(self, frame: np.ndarray) -> None:
        """Non-blocking hand-off. Drops the oldest queued frame instead of
        blocking the caller if the worker is falling behind — the
        navigation loop must never wait on this.
        """
        try:
            self._input_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._input_queue.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            try:
                self._input_queue.put_nowait(frame)
            except queue.Full:
                pass  # extremely unlikely race; just skip this frame

    def _worker_loop(self) -> None:
        min_period = 1.0 / self.inference_frequency_hz
        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                frame = self._input_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                self._run_backend(frame)
            except Exception as exc:  # noqa: BLE001
                logger.error("DA3 inference failed on a frame: %s", exc)

            elapsed = time.time() - loop_start
            sleep_for = min_period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _run_backend(self, frame: np.ndarray) -> np.ndarray:
        """One backend inference + bookkeeping. Shared by the async worker
        and the synchronous path so both record timing identically.
        """
        started = time.perf_counter()
        depth = self._backend.infer(frame)
        self._last_inference_ms = (time.perf_counter() - started) * 1000.0

        confidence = np.ones_like(depth) * (0.5 if self._using_mock else 0.9)
        with self._lock:
            self._latest_depth = depth
            self._latest_confidence = confidence
        return depth

    def infer_sync(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run depth on THIS frame, in the calling thread, and return it.

        Used by runtime/perception_process.py, which must pair a depth map
        with the detections from the same frame — the async path could hand
        back a map from an older frame, which would quietly mis-associate
        depth with the wrong scene.
        """
        if self._backend is None:
            return None
        try:
            return self._run_backend(frame)
        except Exception as exc:  # noqa: BLE001
            logger.error("DA3 synchronous inference failed: %s", exc)
            return None

    def get_depth(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_depth is None else self._latest_depth.copy()

    def get_confidence(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_confidence is None else self._latest_confidence.copy()

    def get_depth_kind(self) -> DepthKind:
        """Monocular Depth Anything output is RELATIVE, and so is the mock's.

        Neither is in meters. Nothing in this project may present these
        numbers as metric distance without first solving for scale against a
        known reference.
        """
        return DepthKind.RELATIVE

    def get_model_info(self) -> dict:
        info = {
            "backend": "mock" if self._using_mock else "transformers",
            "model": "mock-da3" if self._using_mock else self.model_path,
            "device": "cpu" if self._using_mock else self.device,
            "precision": self.precision,
            "inference_resolution": list(self.inference_resolution),
            "depth_kind": self.get_depth_kind().value,
            "simulated": self._using_mock,
        }
        if self._load_error:
            info["load_error"] = self._load_error
        return info

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def is_mock(self) -> bool:
        return self._using_mock


def build_depth_estimator_from_config(cfg: dict) -> BaseDepthEstimator:
    d_cfg = cfg["perception"]["depth"]
    return DA3DepthEstimator(
        model_size=d_cfg.get("model_size", "small"),
        device=d_cfg.get("device", "cpu"),
        precision=d_cfg.get("precision", "fp32"),
        inference_resolution=tuple(d_cfg.get("inference_resolution", [504, 280])),
        inference_frequency_hz=d_cfg.get("inference_frequency_hz", 6.0),
        max_queue_size=d_cfg.get("max_queue_size", 2),
        backend=d_cfg.get("backend", "auto"),
        model_path=d_cfg.get("model_path", ""),
    )
