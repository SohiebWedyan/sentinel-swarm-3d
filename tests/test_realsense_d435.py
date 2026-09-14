"""Hardware-free contract tests for RealSense D435 validation."""
from __future__ import annotations

import numpy as np

from common.config_loader import load_system_config
from hardware.realsense_d435 import D435Frame, RealSenseD435Validator


class _FakeD435Session:
    def __init__(self, cfg, *, usb_type="3.2", name="Intel RealSense D435", pointcloud=True):
        self.i = 0
        self.usb_type, self.name, self.pointcloud = usb_type, name, pointcloud
        self.closed = False

    def open(self):
        return {
            "name": self.name, "usb_type": self.usb_type,
            "color_intrinsics": {"fx": 615.0, "fy": 615.0},
            "depth_intrinsics": {"fx": 385.0, "fy": 385.0},
        }

    def next_frame(self, timeout_ms):
        self.i += 1
        color = np.zeros((4, 6, 3), dtype=np.uint8)
        depth = np.full((4, 6), 1000, dtype=np.uint16)
        points = np.ones((24, 3), dtype=np.float32) if self.pointcloud else np.zeros((24, 3), dtype=np.float32)
        return D435Frame(self.i / 30.0, color, depth, points)

    def close(self):
        self.closed = True


def _clock():
    # Ends the duration condition after the first frame; minimum_frames still
    # drives a deterministic hardware-free sample count.
    values = iter([0.0, 10.0])
    return lambda: next(values, 10.0)


def _validator(**session_kwargs):
    cfg = {"fps": 30, "sample_duration_s": 0.1, "minimum_frames": 4,
           "frame_timeout_ms": 1, "required_usb_prefix": "3.", "minimum_fps_ratio": 0.85}
    return RealSenseD435Validator(cfg, session_factory=lambda c: _FakeD435Session(c, **session_kwargs), clock=_clock())


def _status(report, name):
    return next(check.status for check in report.checks if check.name == name)


def test_mock_d435_passes_rgb_depth_alignment_pointcloud_usb_and_fps():
    report = _validator().run()
    assert report.overall_status == "PASS"
    assert report.sample_count == 4
    for name in ("Device model", "RGB", "Depth", "RGB-Depth alignment", "PointCloud", "USB3", "FPS"):
        assert _status(report, name) == "PASS"
    assert report.rgb_fps == 30.0 and report.depth_fps == 30.0


def test_usb2_is_a_failure_not_a_warning():
    report = _validator(usb_type="2.1").run()
    assert _status(report, "USB3") == "FAIL"
    assert report.overall_status == "FAIL"


def test_d435i_is_explicitly_rejected_so_no_imu_assumption_can_slip_in():
    report = _validator(name="Intel RealSense D435i").run()
    assert _status(report, "Device model") == "FAIL"


def test_empty_pointcloud_fails_validation():
    report = _validator(pointcloud=False).run()
    assert _status(report, "PointCloud") == "FAIL"


def test_low_fps_is_visible_as_warning_without_hiding_other_results():
    class SlowSession(_FakeD435Session):
        def next_frame(self, timeout_ms):
            frame = super().next_frame(timeout_ms)
            return D435Frame(self.i / 5.0, frame.color, frame.depth, frame.points_xyz)
    cfg = {"fps": 30, "sample_duration_s": 0.1, "minimum_frames": 4, "frame_timeout_ms": 1}
    report = RealSenseD435Validator(cfg, session_factory=SlowSession, clock=_clock()).run()
    assert _status(report, "FPS") == "WARN"
    assert report.overall_status == "PASS"


def test_shipped_config_declares_d435_without_enabling_a_d435i_imu_stream():
    cfg = load_system_config()
    d435 = cfg["camera"]["realsense_d435"]
    assert d435["fps"] == 30
    assert d435["warmup_frames"] == 10
    assert d435["required_usb_prefix"] == "3."
    assert "imu" not in d435
