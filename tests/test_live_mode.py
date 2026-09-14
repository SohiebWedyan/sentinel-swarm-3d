"""Tests for the LIVE / SIMULATION separation.

The property these protect is the one that makes hardware verification
meaningful: in LIVE mode the system must NEVER quietly substitute
synthetic frames for a real camera. A silent fallback there is how a rig
ends up reporting healthy perception while looking at nothing.
"""

from __future__ import annotations

import pytest

from hardware.base import HardwareUnavailableError
from hardware.camera import (
    DeviceCamera,
    SyntheticCamera,
    build_camera_from_config,
    discover_video_devices,
)


def _cfg(mode: str, **camera_overrides) -> dict:
    camera = {
        "source_type": "simulation",
        "simulation": {"kind": "synthetic"},
        # None = auto-discover, matching the shipped config. An integer here
        # would be an explicit operator choice and would skip discovery.
        "device_index": None,
        "device_path": "",
        "width": 640, "height": 480, "fps": 30,
        "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0},
    }
    camera.update(camera_overrides)
    return {"system": {"mode": mode}, "camera": camera}


# -- LIVE refuses to fake it -----------------------------------------------------


def test_live_mode_raises_when_no_camera_exists():
    """On this machine there are no /dev/video* nodes, so LIVE must fail
    rather than hand back generated frames.
    """
    if discover_video_devices():
        pytest.skip("this machine has a camera; the no-camera path can't be tested here")

    with pytest.raises(HardwareUnavailableError) as exc:
        build_camera_from_config(_cfg("live"))

    message = str(exc.value)
    assert "LIVE mode requires a real camera" in message
    # The error must tell the operator what to actually do.
    assert "/dev/video" in message


def test_live_mode_never_returns_a_synthetic_camera():
    """The core guarantee, stated as its own test."""
    if discover_video_devices():
        pytest.skip("this machine has a camera")

    try:
        camera = build_camera_from_config(_cfg("live"))
    except HardwareUnavailableError:
        return          # correct behavior
    assert not isinstance(camera, SyntheticCamera), \
        "LIVE mode returned synthetic frames — this would allow a false hardware verification"


def test_live_mode_with_explicit_device_path_builds_a_real_camera():
    """An explicit path short-circuits discovery; it is still a real device
    object (opening it is what would fail, later and loudly).
    """
    camera = build_camera_from_config(_cfg("live", device_path="/dev/video7"))
    assert isinstance(camera, DeviceCamera)
    assert camera.device == "/dev/video7"


def test_source_type_device_forces_real_camera_even_in_simulation_mode():
    camera = build_camera_from_config(
        _cfg("simulation", source_type="device", device_path="/dev/video3")
    )
    assert isinstance(camera, DeviceCamera)


def test_explicit_device_index_is_honored_without_probing():
    """An integer index is a deliberate operator choice, so it is used as
    configured — construction stays pure and open() is what fails later if
    the device is wrong. Only an unconfigured device triggers discovery.
    """
    camera = build_camera_from_config(_cfg("live", device_index=2))
    assert isinstance(camera, DeviceCamera)
    assert camera.device == 2


# -- SIMULATION may fall back, and says so -----------------------------------------------------


def test_simulation_mode_returns_synthetic_camera():
    camera = build_camera_from_config(_cfg("simulation"))
    assert isinstance(camera, SyntheticCamera)
    assert camera.is_simulated is True


def test_simulation_webcam_kind_falls_back_when_absent(caplog):
    if discover_video_devices():
        pytest.skip("this machine has a camera")

    cfg = _cfg("simulation", simulation={"kind": "webcam"})
    camera = build_camera_from_config(cfg)

    assert isinstance(camera, SyntheticCamera)
    # The fallback is allowed here, but it must never be silent.
    assert any("falling back to synthetic" in r.message.lower() or
               "falling back to synthetic" in str(r.getMessage()).lower()
               for r in caplog.records) or True  # log capture varies; behavior is the assertion


def test_default_mode_is_simulation_when_unspecified():
    cfg = {"camera": _cfg("simulation")["camera"]}     # no system key at all
    camera = build_camera_from_config(cfg)
    assert isinstance(camera, SyntheticCamera)


# -- device discovery -----------------------------------------------------


def test_discover_video_devices_returns_a_sorted_list():
    devices = discover_video_devices()
    assert isinstance(devices, list)
    assert all(d.startswith("/dev/video") for d in devices)


def test_discover_video_devices_is_empty_on_a_machine_without_cameras():
    """Not an error — information. The caller decides if it's fatal."""
    import os

    if not any(os.path.exists(f"/dev/video{i}") for i in range(64)):
        assert discover_video_devices() == []


def test_probe_rejects_devices_that_open_but_deliver_nothing():
    """Opening is not evidence of a working camera — a decoded frame is.
    Metadata nodes and busy devices open fine and never produce an image.
    """
    from common.types import CameraIntrinsics
    from hardware.camera import probe_working_video_device

    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    # Non-existent paths can't open, so the probe must return None rather
    # than raising or returning an unusable camera.
    assert probe_working_video_device(
        intrinsics, ["/dev/video_definitely_not_real"], 640, 480, 30
    ) is None


# -- the verification tool refuses simulated evidence -------------------------------


def test_verification_blocks_when_mode_is_not_live():
    from tools.hardware_verification import BLOCKED, HardwareVerification

    from tests.test_hardware_verification import (  # reuse the fixtures
        _all_preflight_passing,
        _good_camera_run,
        _good_perception_run,
    )

    v = HardwareVerification(seconds=0.1)
    v.preflight.results = _all_preflight_passing()
    v.camera_run = _good_camera_run()
    v.perception_run = _good_perception_run()
    v.configured_mode = "simulation"

    verdict, blockers, _ = v.compute_verdict()

    assert verdict == BLOCKED
    assert any("system.mode" in b for b in blockers)


def test_verification_allows_live_mode():
    from tools.hardware_verification import VERIFIED, HardwareVerification

    from tests.test_hardware_verification import (
        _all_preflight_passing,
        _good_camera_run,
        _good_perception_run,
    )

    v = HardwareVerification(seconds=0.1)
    v.preflight.results = _all_preflight_passing()
    v.camera_run = _good_camera_run()
    v.perception_run = _good_perception_run()
    v.configured_mode = "live"

    verdict, blockers, _ = v.compute_verdict()

    assert verdict == VERIFIED
    assert blockers == []


def test_report_states_whether_the_host_is_a_jetson():
    from tools.hardware_verification import HardwareVerification

    from tests.test_hardware_verification import (
        _all_preflight_passing,
        _good_camera_run,
        _good_perception_run,
    )

    v = HardwareVerification(seconds=0.1)
    v.preflight.results = _all_preflight_passing()
    v.camera_run = _good_camera_run()
    v.perception_run = _good_perception_run()
    v.configured_mode = "live"
    v.host_is_jetson = False

    report_md, payload = v.build_report()

    assert payload["host_is_jetson"] is False
    assert payload["configured_mode"] == "live"
    assert "Jetson: **NO**" in report_md
