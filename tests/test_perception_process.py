"""Tests for runtime/perception_process.py and the Phase 2 perception types.

Hardware-free and model-free: a stub detector and the DA3 mock backend
stand in for YOLO and Depth Anything, so synchronization, stale-frame
dropping, timing, degraded-branch behavior and depth-unit honesty are all
verified without weights, CUDA, or a camera.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from common.types import (
    BBox,
    Detection2D,
    DepthFrame,
    DepthKind,
    FramePacket,
    PerceptionHealth,
    PerceptionResult,
)
from perception.depth.da3_depth_estimator import DA3DepthEstimator
from perception.detector.base_detector import BaseObjectDetector
from runtime.perception_process import (
    PerceptionProcessCore,
    build_perception_process_core,
    run_perception_process_entrypoint,
)
from runtime.shared_state import LatestValueQueue


class _StubDetector(BaseObjectDetector):
    """Deterministic stand-in for YOLO."""

    def __init__(self, detections_per_frame: int = 2, degraded: bool = False,
                 raise_on_infer: bool = False) -> None:
        self.detections_per_frame = detections_per_frame
        self._degraded = degraded
        self.raise_on_infer = raise_on_infer
        self.initialized = False
        self.frames_seen = 0

    def initialize(self) -> None:
        self.initialized = True

    def infer(self, frame):
        self.frames_seen += 1
        if self.raise_on_infer:
            raise RuntimeError("detector exploded (test)")
        if self._degraded:
            return []
        return [
            Detection2D(bbox=BBox(10.0 * i, 20.0 * i, 10.0 * i + 30, 20.0 * i + 40),
                        class_id=i, class_name=f"class_{i}", confidence=0.9 - 0.1 * i)
            for i in range(self.detections_per_frame)
        ]

    def shutdown(self) -> None:
        self.initialized = False

    @property
    def is_degraded(self) -> bool:
        return self._degraded

    def get_model_info(self) -> dict:
        return {"model_path": "stub", "device": "cpu", "degraded": self._degraded}


def _frame(sequence_id: int = 1, age_s: float = 0.0, size=(240, 320)) -> FramePacket:
    image = np.random.randint(0, 255, size=(size[0], size[1], 3), dtype=np.uint8)
    return FramePacket(
        image=image,
        timestamp=time.time() - age_s,
        sequence_id=sequence_id,
        frame_id="camera",
        width=size[1],
        height=size[0],
    )


def _core(**kwargs) -> PerceptionProcessCore:
    depth = kwargs.pop("depth_estimator", None)
    if depth is None and kwargs.pop("with_depth", True):
        depth = DA3DepthEstimator(inference_resolution=(64, 48), backend="mock")
    return PerceptionProcessCore(
        detector=kwargs.pop("detector", _StubDetector()),
        depth_estimator=depth,
        **kwargs,
    )


def _wait_for(predicate, timeout_s: float = 8.0, interval_s: float = 0.05) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


# -- synchronized output -----------------------------------------------------


def test_result_pairs_detections_and_depth_from_the_same_frame():
    core = _core()
    core.start()
    try:
        packet = _frame(sequence_id=42)
        result = core.process_frame(packet)

        assert isinstance(result, PerceptionResult)
        # The identity of the source frame is preserved on the result, which
        # is what makes the pairing verifiable downstream.
        assert result.sequence_id == 42
        assert result.timestamp == packet.timestamp
        assert result.detection_count == 2
        assert result.has_depth is True
    finally:
        core.stop()


def test_result_carries_image_dimensions_and_optional_frame():
    core = _core(publish_frame=True)
    core.start()
    try:
        result = core.process_frame(_frame(size=(240, 320)))
        assert result.image_width == 320
        assert result.image_height == 240
        assert result.image is not None
    finally:
        core.stop()


def test_frame_can_be_withheld_from_the_result_to_keep_ipc_cheap():
    core = _core(publish_frame=False)
    core.start()
    try:
        result = core.process_frame(_frame())
        assert result.image is None
        assert result.detection_count == 2   # everything else still present
    finally:
        core.stop()


def test_depth_can_be_withheld_from_the_result():
    core = _core(publish_depth=False)
    core.start()
    try:
        result = core.process_frame(_frame())
        assert result.depth is None
    finally:
        core.stop()


def test_detections_expose_center_and_size_accessors():
    core = _core()
    core.start()
    try:
        result = core.process_frame(_frame())
        det = result.detections[0]
        assert det.width == det.x2 - det.x1
        assert det.height == det.y2 - det.y1
        assert det.center_x == (det.x1 + det.x2) / 2
        assert det.center_y == (det.y1 + det.y2) / 2
        assert set(det.to_dict()) >= {
            "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2",
            "center_x", "center_y", "width", "height",
        }
    finally:
        core.stop()


# -- stale frames -----------------------------------------------------


def test_stale_frames_are_dropped_not_processed():
    core = _core(stale_frame_max_age_s=0.1)
    core.start()
    try:
        result = core.process_frame(_frame(age_s=1.0))
        assert result is None
        assert core.get_health().frames_dropped_stale == 1
        # The detector must not have been asked to work on it at all.
        assert core.detector.frames_seen == 0
    finally:
        core.stop()


def test_fresh_frames_are_processed():
    core = _core(stale_frame_max_age_s=1.0)
    core.start()
    try:
        assert core.process_frame(_frame(age_s=0.0)) is not None
        assert core.get_health().frames_dropped_stale == 0
    finally:
        core.stop()


# -- timing -----------------------------------------------------


def test_timing_is_populated_for_each_stage():
    core = _core()
    core.start()
    try:
        result = core.process_frame(_frame())
        timing = result.timing

        assert timing.detector_ms >= 0.0
        assert timing.depth_ms > 0.0
        assert timing.total_ms >= 0.0
        assert timing.pipeline_latency_ms >= 0.0
    finally:
        core.stop()


def test_pipeline_latency_is_measured_from_capture_not_from_inference_start():
    """Latency must describe how old the information is, which is what a
    consumer actually cares about.
    """
    core = _core(stale_frame_max_age_s=10.0)
    core.start()
    try:
        result = core.process_frame(_frame(age_s=0.4))
        assert result.timing.pipeline_latency_ms >= 400.0
    finally:
        core.stop()


# -- degraded branches -----------------------------------------------------


def test_degraded_detector_still_produces_a_result_but_is_not_healthy():
    core = _core(detector=_StubDetector(degraded=True))
    core.start()
    try:
        result = core.process_frame(_frame())
        assert result is not None
        assert result.detection_count == 0
        assert result.has_depth is True      # the other branch still works

        health = core.get_health()
        assert health.detector_degraded is True
        assert health.detector_ready is False
        # A process producing empty results must never report itself healthy.
        assert health.healthy is False
    finally:
        core.stop()


def test_detector_exception_degrades_only_that_branch():
    core = _core(detector=_StubDetector(raise_on_infer=True))
    core.start()
    try:
        result = core.process_frame(_frame())
        assert result is not None
        assert result.detection_count == 0
        assert result.has_depth is True
        assert "detector failed" in (core.get_health().error_message or "")
    finally:
        core.stop()


def test_depth_disabled_still_produces_detections():
    core = _core(with_depth=False, depth_enabled=False)
    core.start()
    try:
        result = core.process_frame(_frame())
        assert result.detection_count == 2
        assert result.depth is None
        assert result.timing.depth_ms == 0.0
    finally:
        core.stop()


# -- depth semantics (honest units) -----------------------------------------------------


def test_depth_is_labelled_relative_not_metric():
    core = _core()
    core.start()
    try:
        result = core.process_frame(_frame())
        depth: DepthFrame = result.depth

        assert depth.kind is DepthKind.RELATIVE
        assert depth.is_metric is False
        assert "meters" not in depth.describe_units()
    finally:
        core.stop()


def test_mock_depth_is_flagged_as_simulated():
    core = _core()
    core.start()
    try:
        result = core.process_frame(_frame())
        assert result.depth.simulated is True
        assert core.get_health().depth_simulated is True
    finally:
        core.stop()


def test_depth_estimator_reports_relative_kind_and_model_info():
    estimator = DA3DepthEstimator(inference_resolution=(32, 24), backend="mock")
    estimator.initialize()
    try:
        assert estimator.get_depth_kind() is DepthKind.RELATIVE
        info = estimator.get_model_info()
        assert info["backend"] == "mock"
        assert info["simulated"] is True
        assert info["depth_kind"] == "relative"
    finally:
        estimator.shutdown()


def test_infer_sync_returns_this_frames_depth():
    """The synchronous path is what makes detection/depth pairing correct."""
    estimator = DA3DepthEstimator(inference_resolution=(32, 24), backend="mock")
    estimator.initialize()
    try:
        frame = np.random.randint(0, 255, size=(100, 120, 3), dtype=np.uint8)
        depth = estimator.infer_sync(frame)

        assert depth is not None
        assert depth.shape == (24, 32)
        assert estimator.last_inference_ms > 0.0
    finally:
        estimator.shutdown()


def test_unknown_depth_kind_is_the_safe_default_on_the_base_interface():
    """An estimator that hasn't declared units must never be assumed metric."""
    from perception.depth.base_depth_estimator import BaseDepthEstimator

    class _Bare(BaseDepthEstimator):
        def initialize(self): ...
        def infer(self, frame): ...
        def get_depth(self): return None
        def get_confidence(self): return None
        def shutdown(self): ...

    assert _Bare().get_depth_kind() is DepthKind.UNKNOWN


# -- health -----------------------------------------------------


def test_health_becomes_healthy_once_results_flow():
    core = _core()
    core.start()
    try:
        assert core.is_healthy() is False    # nothing processed yet
        core.process_frame(_frame())
        health: PerceptionHealth = core.get_health()

        assert health.healthy is True
        assert health.frames_processed == 1
        assert health.depth_ready is True
        assert health.to_dict()["depth_kind"] == "relative"
    finally:
        core.stop()


def test_fps_is_measured_across_frames():
    core = _core()
    core.start()
    try:
        for seq in range(3):
            core.process_frame(_frame(sequence_id=seq))
            time.sleep(0.02)
        assert core.get_health().fps > 0.0
    finally:
        core.stop()


def test_build_from_config_wires_the_process():
    cfg = {
        "perception": {
            "detector": {"model_path": "yolov8n.pt", "device": "cpu", "image_size": 320},
            "depth": {"enabled": True, "backend": "mock", "inference_resolution": [32, 24]},
            "process": {"stale_frame_max_age_s": 0.25, "publish_frame": False,
                        "publish_depth": True},
        }
    }

    core = build_perception_process_core(cfg)

    assert core.stale_frame_max_age_s == 0.25
    assert core.publish_frame is False
    assert core.depth_enabled is True


def test_depth_can_be_disabled_from_config():
    cfg = {
        "perception": {
            "detector": {"model_path": "yolov8n.pt", "device": "cpu"},
            "depth": {"enabled": False},
            "process": {},
        }
    }

    core = build_perception_process_core(cfg)

    assert core.depth_enabled is False
    assert core.depth_estimator is None


# -- process entrypoint -----------------------------------------------------


def test_entrypoint_reports_a_build_failure_instead_of_crashing():
    """A bad config must surface as unhealthy with an explanation, not as an
    exception that takes the leader runtime down.
    """
    cfg = {
        "system": {"log_level": "WARNING"},
        "robots": {"this_robot": "leader"},
        # perception key missing entirely -> build fails
    }

    frame_input: LatestValueQueue = LatestValueQueue()
    result_output: LatestValueQueue = LatestValueQueue()
    health_output: LatestValueQueue = LatestValueQueue()
    stop_event = mp.Event()
    stop_event.set()

    run_perception_process_entrypoint(cfg, frame_input, result_output, health_output, stop_event)

    health = health_output.get_latest()
    assert health is not None
    assert health.healthy is False
    assert health.error_message
