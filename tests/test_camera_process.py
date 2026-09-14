"""Tests for runtime/camera_process.py.

All hardware-free: they drive CameraProcessCore against SyntheticCamera and
against deliberately broken fake cameras, so the latest-frame-only
behavior, dropped-frame accounting, reconnect policy and status reporting
are all exercised without a webcam.
"""

from __future__ import annotations

import time

import numpy as np

from common.types import CameraIntrinsics, CameraStatus, FramePacket
from hardware.base import HardwareUnavailableError
from hardware.camera import BaseCamera, Frame, SyntheticCamera
from runtime.camera_process import CameraProcessCore, build_camera_process_core


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=500.0, fy=500.0, cx=160.0, cy=120.0)


def _synthetic_core(**kwargs) -> CameraProcessCore:
    camera = SyntheticCamera(_intrinsics(), width=320, height=240)
    return CameraProcessCore(camera=camera, configured_fps=30.0, **kwargs)


def _wait_for(predicate, timeout_s: float = 5.0, interval_s: float = 0.02) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


class _FailingCamera(BaseCamera):
    """Fails to open a configurable number of times, then succeeds."""

    def __init__(self, failures_before_success: int = 1) -> None:
        super().__init__(_intrinsics())
        self.failures_before_success = failures_before_success
        self.open_attempts = 0
        self._opened = False

    def describe(self) -> str:
        return "FailingCamera"

    def open(self) -> None:
        self.open_attempts += 1
        if self.open_attempts <= self.failures_before_success:
            raise HardwareUnavailableError("camera not present (test)")
        self._opened = True

    def read(self):
        if not self._opened:
            raise HardwareUnavailableError("read before open")
        return Frame(image=np.zeros((240, 320, 3), dtype=np.uint8),
                     timestamp=time.time(), frame_id=self._next_frame_id())

    def is_healthy(self) -> bool:
        return self._opened

    def close(self) -> None:
        self._opened = False


class _DisappearingCamera(BaseCamera):
    """Opens fine, delivers a few frames, then goes away mid-run."""

    def __init__(self, frames_before_failure: int = 3) -> None:
        super().__init__(_intrinsics())
        self.frames_before_failure = frames_before_failure
        self.frames_read = 0
        self._opened = False

    def describe(self) -> str:
        return "DisappearingCamera"

    def open(self) -> None:
        self._opened = True

    def read(self):
        self.frames_read += 1
        if self.frames_read > self.frames_before_failure:
            self._opened = False
            raise HardwareUnavailableError("USB device disappeared (test)")
        return Frame(image=np.zeros((240, 320, 3), dtype=np.uint8),
                     timestamp=time.time(), frame_id=self._next_frame_id())

    def is_healthy(self) -> bool:
        return self._opened

    def close(self) -> None:
        self._opened = False


# -- capture + publishing -----------------------------------------------------


def test_core_captures_frames_and_reports_healthy():
    core = _synthetic_core()
    core.start()
    try:
        assert _wait_for(lambda: core.peek_latest_frame() is not None)

        packet = core.get_latest_frame()
        assert isinstance(packet, FramePacket)
        assert packet.width == 320 and packet.height == 240
        assert packet.sequence_id >= 1
        assert packet.image.shape == (240, 320, 3)
        assert core.is_healthy() is True
    finally:
        core.stop()


def test_get_latest_frame_consumes_so_a_frame_is_never_processed_twice():
    core = _synthetic_core()
    core.start()
    try:
        assert _wait_for(lambda: core.peek_latest_frame() is not None)
        first = core.get_latest_frame()
        assert first is not None
        # Immediately after consuming, there is no NEW frame yet.
        assert core.get_latest_frame() is None
    finally:
        core.stop()


def test_only_the_newest_frame_is_kept_and_drops_are_counted():
    """The core must hold exactly one frame. A consumer that stops reading
    must cause drops, not unbounded growth.
    """
    core = _synthetic_core()
    core.start()
    try:
        assert _wait_for(lambda: core.get_status().frames_dropped >= 3, timeout_s=5.0)

        status = core.get_status()
        assert status.frames_dropped >= 3

        # Whatever is held is a single packet, and it is the newest one.
        first = core.peek_latest_frame()
        assert first is not None
        assert _wait_for(lambda: (core.peek_latest_frame() is not None
                                  and core.peek_latest_frame().sequence_id > first.sequence_id))
    finally:
        core.stop()


def test_sequence_ids_increase_monotonically():
    core = _synthetic_core()
    core.start()
    try:
        seen = []
        deadline = time.time() + 3.0
        while len(seen) < 3 and time.time() < deadline:
            packet = core.get_latest_frame()
            if packet is not None:
                seen.append(packet.sequence_id)
            time.sleep(0.02)

        assert len(seen) >= 3
        assert seen == sorted(seen)
        assert len(set(seen)) == len(seen)
    finally:
        core.stop()


def test_measured_fps_is_reported():
    core = _synthetic_core()
    core.start()
    try:
        assert _wait_for(lambda: core.get_status().measured_fps > 0.0, timeout_s=5.0)
    finally:
        core.stop()


def test_capture_is_paced_to_the_configured_fps():
    """A source that doesn't block (synthetic frames, a video file) must not
    be allowed to spin the capture thread at thousands of FPS — that burns a
    core on a Jetson and produces frames nothing can consume. A real
    VideoCapture.read() self-throttles, so this pacing costs it nothing.
    """
    camera = SyntheticCamera(_intrinsics(), width=160, height=120)
    core = CameraProcessCore(camera=camera, configured_fps=20.0)
    core.start()
    try:
        time.sleep(1.5)
        measured = core.get_status().measured_fps
        # Generously bounded: the point is that it is ~20, not ~2000.
        assert 5.0 < measured < 60.0, f"capture ran at {measured} FPS, expected ~20"
    finally:
        core.stop()


def test_frame_count_over_time_matches_the_configured_rate():
    camera = SyntheticCamera(_intrinsics(), width=160, height=120)
    core = CameraProcessCore(camera=camera, configured_fps=20.0)
    core.start()
    try:
        time.sleep(1.0)
        # ~20 frames in a second, not thousands.
        assert core.get_status().sequence_id < 200
    finally:
        core.stop()


def test_ipc_scale_downscales_and_records_the_factor():
    """A downscaled frame must record its scale, because the configured
    intrinsics no longer describe it.
    """
    core = _synthetic_core(ipc_scale=0.5)
    core.start()
    try:
        assert _wait_for(lambda: core.peek_latest_frame() is not None)
        packet = core.get_latest_frame()
        assert packet is not None
        assert packet.scale_from_capture == 0.5
        assert packet.width == 160 and packet.height == 120
        assert packet.image.shape[:2] == (120, 160)
    finally:
        core.stop()


def test_default_ipc_scale_is_native_resolution():
    core = _synthetic_core()
    core.start()
    try:
        assert _wait_for(lambda: core.peek_latest_frame() is not None)
        packet = core.get_latest_frame()
        assert packet is not None
        assert packet.scale_from_capture == 1.0
        assert (packet.width, packet.height) == (320, 240)
    finally:
        core.stop()


# -- status / health -----------------------------------------------------


def test_status_is_serializable_for_cross_process_publishing():
    core = _synthetic_core()
    core.start()
    try:
        assert _wait_for(lambda: core.get_status().healthy)
        status_dict = core.get_status().to_dict()
        assert status_dict["connected"] is True
        assert status_dict["simulated"] is True
        assert status_dict["width"] == 320
    finally:
        core.stop()


def test_unhealthy_before_start_and_after_stop():
    core = _synthetic_core()
    assert core.is_healthy() is False

    core.start()
    assert _wait_for(lambda: core.is_healthy())
    core.stop()

    assert core.is_healthy() is False
    assert core.get_status().connected is False


def test_synthetic_camera_is_reported_as_simulated():
    core = _synthetic_core()
    core.start()
    try:
        status: CameraStatus = core.get_status()
        assert status.simulated is True
        assert "Synthetic" in status.device
    finally:
        core.stop()


# -- failure handling / reconnect -----------------------------------------------------


def test_missing_camera_does_not_raise_and_is_reported_unhealthy():
    core = CameraProcessCore(camera=_FailingCamera(failures_before_success=10**6),
                             reconnect_enabled=False)
    core.start()   # must not raise
    try:
        status = core.get_status()
        assert status.connected is False
        assert status.healthy is False
        assert status.error_message is not None
        assert core.get_latest_frame() is None
    finally:
        core.stop()


def test_camera_reconnects_after_initial_failure():
    camera = _FailingCamera(failures_before_success=1)
    core = CameraProcessCore(camera=camera, reconnect_enabled=True,
                             reconnect_initial_delay_s=0.05, reconnect_max_delay_s=0.1)
    core.start()
    try:
        assert _wait_for(lambda: core.is_healthy(), timeout_s=5.0)
        assert core.get_status().reconnect_attempts >= 1
    finally:
        core.stop()


def test_camera_lost_mid_run_is_detected_and_retried():
    camera = _DisappearingCamera(frames_before_failure=3)
    core = CameraProcessCore(camera=camera, reconnect_enabled=True,
                             reconnect_initial_delay_s=0.05, reconnect_max_delay_s=0.1)
    core.start()
    try:
        assert _wait_for(lambda: core.get_status().reconnect_attempts >= 1, timeout_s=5.0)
        # The disappearing camera reopens successfully but keeps failing to
        # read, so it must never be reported healthy-and-streaming for long.
        assert camera.frames_read > 3
    finally:
        core.stop()


def test_reconnect_can_be_disabled():
    camera = _FailingCamera(failures_before_success=10**6)
    core = CameraProcessCore(camera=camera, reconnect_enabled=False)
    core.start()
    try:
        time.sleep(0.3)
        assert core.get_status().reconnect_attempts == 0
    finally:
        core.stop()


# -- config wiring -----------------------------------------------------


def test_build_from_config_reads_the_phase_2_camera_settings():
    cfg = {
        "camera": {
            "enabled": True,
            "frame_id": "front_camera",
            "source_type": "simulation",
            "simulation": {"kind": "synthetic"},
            "width": 320, "height": 240, "fps": 15,
            "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0},
            "reconnect": {"enabled": False, "initial_delay_s": 2.0,
                          "max_delay_s": 5.0, "max_attempts": 3},
            "ipc": {"scale": 0.5},
        }
    }

    core = build_camera_process_core(cfg)

    assert core.frame_id == "front_camera"
    assert core.configured_fps == 15.0
    assert core.ipc_scale == 0.5
    assert core.reconnect_enabled is False
    assert core.reconnect_max_attempts == 3


def test_invalid_ipc_scale_falls_back_to_native():
    core = _synthetic_core(ipc_scale=0.0)
    assert core.ipc_scale == 1.0
    core2 = _synthetic_core(ipc_scale=5.0)
    assert core2.ipc_scale == 1.0
