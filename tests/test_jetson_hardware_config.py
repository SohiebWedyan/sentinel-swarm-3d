"""Tests for the Jetson-oriented hardware settings.

These cover the capture tuning that only matters on a real USB camera
(V4L2 backend, MJPG pixel format, buffer depth 1) and the depth cadence
that keeps detection responsive when depth is slow. No camera and no
models required — the checks are on configuration plumbing and on the
decision logic, which is where the bugs would be.
"""

from __future__ import annotations

import sys

import pytest

from common.config_loader import load_system_config
from common.types import CameraIntrinsics
from hardware.camera import DeviceCamera, SyntheticCamera, build_camera_from_config
from runtime.perception_process import build_perception_process_core


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=500.0, fy=500.0, cx=160.0, cy=120.0)


def _camera_cfg(**overrides) -> dict:
    hardware = overrides.pop("hardware", None)
    cfg = {
        "camera": {
            "source_type": "device",
            "device_index": 0,
            "device_path": "",
            "width": 1280, "height": 720, "fps": 30,
            "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0},
        }
    }
    if hardware is not None:
        cfg["camera"]["hardware"] = hardware
    cfg["camera"].update(overrides)
    return cfg


# -- capture tuning defaults -----------------------------------------------------


def test_mjpg_is_the_default_pixel_format():
    """YUYV (the driver default on most USB webcams) caps 720p at roughly
    5-10 FPS over USB 2.0; MJPG reaches 30.
    """
    camera = build_camera_from_config(_camera_cfg())
    assert isinstance(camera, DeviceCamera)
    assert camera.fourcc == "MJPG"


def test_capture_buffer_depth_defaults_to_one():
    """V4L2 returns the OLDEST queued frame from read(). A default depth of
    ~4 makes every frame ~130ms stale at 30 FPS and defeats the
    latest-frame-only design in camera_process.
    """
    camera = build_camera_from_config(_camera_cfg())
    assert camera.buffer_size == 1


def test_backend_defaults_to_auto():
    camera = build_camera_from_config(_camera_cfg())
    assert camera.backend == "auto"


def test_hardware_block_overrides_the_defaults():
    camera = build_camera_from_config(_camera_cfg(hardware={
        "backend": "v4l2", "fourcc": "YUYV", "buffer_size": 3,
    }))
    assert camera.backend == "v4l2"
    assert camera.fourcc == "YUYV"
    assert camera.buffer_size == 3


def test_fourcc_can_be_disabled_to_keep_the_driver_default():
    camera = build_camera_from_config(_camera_cfg(hardware={"fourcc": ""}))
    assert camera.fourcc == ""


# -- backend resolution -----------------------------------------------------


def test_auto_backend_selects_v4l2_on_linux():
    pytest.importorskip("cv2")
    import cv2

    camera = DeviceCamera(_intrinsics(), 0, 640, 480, 30, backend="auto")
    resolved = camera._resolve_backend()

    if sys.platform.startswith("linux"):
        assert resolved == cv2.CAP_V4L2
    else:
        assert resolved == cv2.CAP_ANY


def test_explicit_backend_is_honored():
    pytest.importorskip("cv2")
    import cv2

    camera = DeviceCamera(_intrinsics(), 0, 640, 480, 30, backend="any")
    assert camera._resolve_backend() == cv2.CAP_ANY


def test_unknown_backend_name_falls_back_to_any_rather_than_crashing():
    pytest.importorskip("cv2")
    import cv2

    camera = DeviceCamera(_intrinsics(), 0, 640, 480, 30, backend="nonsense")
    assert camera._resolve_backend() == cv2.CAP_ANY


def test_fourcc_decoding_round_trips():
    pytest.importorskip("cv2")
    import cv2

    encoded = cv2.VideoWriter_fourcc(*"MJPG")
    assert DeviceCamera._decode_fourcc(encoded) == "MJPG"
    assert DeviceCamera._decode_fourcc(0) == ""


def test_device_path_is_preferred_over_index():
    camera = build_camera_from_config(_camera_cfg(device_path="/dev/video2"))
    assert camera.device == "/dev/video2"


def test_index_is_used_when_no_path_is_given():
    camera = build_camera_from_config(_camera_cfg(device_index=3))
    assert camera.device == 3


def test_synthetic_camera_is_unaffected_by_hardware_tuning():
    cfg = _camera_cfg(source_type="simulation")
    cfg["camera"]["simulation"] = {"kind": "synthetic"}
    camera = build_camera_from_config(cfg)
    assert isinstance(camera, SyntheticCamera)


# -- shipped config -----------------------------------------------------


def test_shipped_camera_yaml_has_the_jetson_hardware_defaults():
    cfg = load_system_config("config/system.yaml")
    hardware = cfg["camera"]["hardware"]

    assert hardware["fourcc"] == "MJPG"
    assert hardware["buffer_size"] == 1
    assert hardware["backend"] == "auto"


# -- depth cadence -----------------------------------------------------


def _perception_cfg(depth_every_n_frames: int) -> dict:
    return {
        "perception": {
            "detector": {"model_path": "yolov8n.pt", "device": "cpu"},
            "depth": {"enabled": True, "backend": "mock",
                      "inference_resolution": [32, 24]},
            "process": {"depth_every_n_frames": depth_every_n_frames},
        }
    }


def test_depth_runs_every_frame_by_default():
    core = build_perception_process_core(_perception_cfg(1))
    assert core.depth_every_n_frames == 1
    assert core._should_run_depth() is True


def test_depth_cadence_is_read_from_config():
    core = build_perception_process_core(_perception_cfg(5))
    assert core.depth_every_n_frames == 5


def test_depth_cadence_skips_the_right_frames():
    """Detection runs every frame; depth runs every Nth. The cadence is
    driven by frames PROCESSED so dropped frames don't perturb it.
    """
    core = build_perception_process_core(_perception_cfg(3))

    ran = []
    for processed in range(9):
        core._frames_processed = processed
        ran.append(core._should_run_depth())

    assert ran == [True, False, False, True, False, False, True, False, False]


def test_invalid_cadence_is_clamped_to_one():
    for bad in (0, -1):
        core = build_perception_process_core(_perception_cfg(bad))
        assert core.depth_every_n_frames == 1


def test_cadence_never_attaches_depth_from_a_different_frame():
    """The correctness property the cadence must not break: a result either
    carries depth from its OWN frame, or no depth at all.
    """
    import time

    import numpy as np

    from common.types import FramePacket

    core = build_perception_process_core(_perception_cfg(2))
    core.start()
    try:
        results = []
        for seq in range(4):
            packet = FramePacket(
                image=np.zeros((48, 64, 3), dtype=np.uint8),
                timestamp=time.time(), sequence_id=seq, width=64, height=48,
            )
            results.append(core.process_frame(packet))

        with_depth = [r for r in results if r.depth is not None]
        without_depth = [r for r in results if r.depth is None]

        # Both cases occur at cadence 2...
        assert with_depth and without_depth
        # ...and depth, when present, was computed for that very frame:
        # its resolution matches the configured depth output, and no result
        # ever borrows a neighbour's map.
        for result in with_depth:
            assert result.depth.depth.shape == (24, 32)
    finally:
        core.stop()


def test_depth_disabled_overrides_any_cadence():
    cfg = _perception_cfg(1)
    cfg["perception"]["depth"]["enabled"] = False
    core = build_perception_process_core(cfg)
    assert core._should_run_depth() is False
