"""Tests for tools/hardware_verification.py's verdict logic.

The verdict is the whole point of that tool — it is what decides whether
Phase 2 gets called HARDWARE VERIFIED. So the rules are tested directly,
with synthetic inputs, rather than trusting a single run on one machine.

The most important property under test: the tool must NEVER report
VERIFIED while running on simulated inputs.
"""

from __future__ import annotations

from tools.hardware_verification import (
    BLOCKED,
    VERIFIED,
    HardwareVerification,
    SustainedResult,
)
from tools.jetson_preflight import PASS, WARN, CheckResult


def _verification() -> HardwareVerification:
    return HardwareVerification(seconds=0.1)


def _all_preflight_passing() -> list[CheckResult]:
    return [
        CheckResult("Platform", PASS),
        CheckResult("Power mode", PASS),
        CheckResult("PyTorch", PASS),
        CheckResult("Ultralytics YOLO", PASS),
        CheckResult("Transformers (depth)", PASS),
        CheckResult("OpenCV", PASS),
        CheckResult("Video devices", PASS),
        CheckResult("Camera capture", PASS),
        CheckResult("YOLO inference", PASS),
        CheckResult("Depth inference", PASS),
        CheckResult("LiDAR port", PASS),
    ]


def _good_camera_run() -> SustainedResult:
    return SustainedResult(
        ran=True, duration_s=15.0, frames=440, fps=29.3,
        extra={"simulated": False, "configured_fps": 30.0, "device": "DeviceCamera(/dev/video0)",
               "width": 1280, "height": 720, "frames_dropped": 2, "healthy": True},
    )


def _good_perception_run() -> SustainedResult:
    return SustainedResult(
        ran=True, duration_s=15.0, frames=120, fps=8.0,
        extra={"detector_degraded": False, "depth_simulated": False,
               "total_detections": 340, "latency_ms_median": 180.0,
               "depth_kind": "relative", "healthy": True},
    )


def _fully_verified() -> HardwareVerification:
    v = _verification()
    v.preflight.results = _all_preflight_passing()
    v.camera_run = _good_camera_run()
    v.perception_run = _good_perception_run()
    # A genuinely verified run is by definition a live-mode run; verification
    # against simulated inputs is itself a blocker (see
    # test_verification_blocks_when_mode_is_not_live in test_live_mode.py).
    v.configured_mode = "live"
    v.host_is_jetson = True
    return v


# -- the happy path -----------------------------------------------------


def test_all_criteria_met_yields_verified():
    verdict, blockers, _ = _fully_verified().compute_verdict()
    assert verdict == VERIFIED
    assert blockers == []


# -- each blocker, individually -----------------------------------------------------


def test_missing_cuda_pytorch_blocks():
    v = _fully_verified()
    v.preflight.results = [r for r in v.preflight.results if r.name != "PyTorch"]
    v.preflight.results.append(CheckResult("PyTorch", "FAIL"))

    verdict, blockers, _ = v.compute_verdict()
    assert verdict == BLOCKED
    assert any("PyTorch" in b for b in blockers)


def test_missing_ultralytics_blocks():
    v = _fully_verified()
    v.preflight.results = [r for r in v.preflight.results if r.name != "Ultralytics YOLO"]
    v.preflight.results.append(CheckResult("Ultralytics YOLO", "FAIL"))

    verdict, blockers, _ = v.compute_verdict()
    assert verdict == BLOCKED
    assert any("Ultralytics" in b for b in blockers)


def test_synthetic_camera_blocks_verification():
    """The single most important guard: simulated inputs must never be
    reported as hardware-verified.
    """
    v = _fully_verified()
    v.camera_run.extra["simulated"] = True

    verdict, blockers, _ = v.compute_verdict()
    assert verdict == BLOCKED
    assert any("SYNTHETIC" in b for b in blockers)


def test_mock_depth_blocks_verification():
    v = _fully_verified()
    v.perception_run.extra["depth_simulated"] = True

    verdict, blockers, _ = v.compute_verdict()
    assert verdict == BLOCKED
    assert any("MOCK" in b for b in blockers)


def test_degraded_detector_blocks():
    v = _fully_verified()
    v.perception_run.extra["detector_degraded"] = True

    verdict, blockers, _ = v.compute_verdict()
    assert verdict == BLOCKED
    assert any("degraded" in b.lower() for b in blockers)


def test_camera_that_never_ran_blocks():
    v = _fully_verified()
    v.camera_run = SustainedResult(ran=False, error="device busy")

    verdict, blockers, _ = v.compute_verdict()
    assert verdict == BLOCKED
    assert any("device busy" in b for b in blockers)


def test_zero_frames_blocks():
    v = _fully_verified()
    v.camera_run.frames = 0

    verdict, blockers, _ = v.compute_verdict()
    assert verdict == BLOCKED
    assert any("zero frames" in b.lower() for b in blockers)


def test_perception_that_never_ran_blocks():
    v = _fully_verified()
    v.perception_run = SustainedResult(ran=False, error="model init failed")

    verdict, blockers, _ = v.compute_verdict()
    assert verdict == BLOCKED
    assert any("model init failed" in b for b in blockers)


# -- warnings must not block -----------------------------------------------------


def test_lidar_problems_do_not_block_phase_2():
    """Phase 2 perception does not depend on the LiDAR, and its runtime
    issue is tracked separately — conflating them would be wrong.
    """
    v = _fully_verified()
    v.preflight.results = [r for r in v.preflight.results if r.name != "LiDAR port"]
    v.preflight.results.append(CheckResult("LiDAR port", "FAIL"))

    verdict, blockers, warnings = v.compute_verdict()
    assert verdict == VERIFIED
    assert blockers == []
    assert any("LiDAR" in w for w in warnings)


def test_low_camera_fps_warns_but_does_not_block():
    v = _fully_verified()
    v.camera_run.fps = 8.0        # against a configured 30

    verdict, blockers, warnings = v.compute_verdict()
    assert verdict == VERIFIED
    assert any("pixel format" in w for w in warnings)


def test_yolo_on_cpu_warns_but_does_not_block():
    v = _fully_verified()
    v.preflight.results = [r for r in v.preflight.results if r.name != "YOLO inference"]
    v.preflight.results.append(CheckResult("YOLO inference", WARN))

    verdict, _, warnings = v.compute_verdict()
    assert verdict == VERIFIED
    assert any("CPU" in w for w in warnings)


def test_high_latency_warns_but_does_not_block():
    v = _fully_verified()
    v.perception_run.extra["latency_ms_median"] = 900.0

    verdict, blockers, warnings = v.compute_verdict()
    assert verdict == VERIFIED
    assert blockers == []
    assert any("latency" in w.lower() for w in warnings)


def test_real_detector_finding_nothing_warns():
    """A real model that detected nothing all run is suspicious enough to
    flag — the camera may be pointed at a blank wall.
    """
    v = _fully_verified()
    v.perception_run.extra["total_detections"] = 0

    verdict, blockers, warnings = v.compute_verdict()
    assert verdict == VERIFIED
    assert blockers == []
    assert any("found nothing" in w for w in warnings)


# -- report generation -----------------------------------------------------


def test_report_contains_the_verdict_and_criteria():
    report_md, payload = _fully_verified().build_report()

    assert VERIFIED in report_md
    assert "Criteria used" in report_md
    assert payload["verdict"] == VERIFIED
    assert payload["blockers"] == []


def test_report_json_is_diffable_between_runs():
    """The JSON must carry the fields a comparison would rely on."""
    _, payload = _fully_verified().build_report()

    for key in ("verdict", "blockers", "warnings", "preflight",
                "camera_sustained", "perception_sustained"):
        assert key in payload

    assert payload["perception_sustained"]["fps"] == 8.0
    assert payload["camera_sustained"]["extra"]["simulated"] is False


def test_blocked_report_lists_every_blocker():
    v = _fully_verified()
    v.camera_run.extra["simulated"] = True
    v.perception_run.extra["depth_simulated"] = True

    report_md, payload = v.build_report()

    assert BLOCKED in report_md
    assert len(payload["blockers"]) == 2
    for blocker in payload["blockers"]:
        assert blocker in report_md


def test_report_repeats_the_relative_depth_caveat():
    """A verified run must not be mistaken for metric depth."""
    report_md, _ = _fully_verified().build_report()
    assert "RELATIVE" in report_md
    assert "metric" in report_md.lower()
