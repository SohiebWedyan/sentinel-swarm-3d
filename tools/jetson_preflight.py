#!/usr/bin/env python3
"""Jetson bring-up preflight check.

Run this ON the Jetson before trying to run the perception pipeline. It
answers, in order, the questions that actually block bring-up:

    Is this a Jetson, and is it in a performance power mode?
    Is torch the real CUDA build, or the wrong wheel?
    Are ultralytics / transformers importable?
    Which /dev/video* devices exist, and what formats do they support?
    Does the configured camera actually open, and at what real FPS?
    Does YOLO load and run, and how long does one frame take?
    Does the depth model load and run, and how long does it take?
    Is the LiDAR serial port present?

Every check reports PASS / WARN / FAIL with a specific next action, and the
exit code is non-zero if anything FAILed — so it can gate a bring-up script.

Nothing here moves a motor, and nothing here writes to the robot.

    python tools/jetson_preflight.py
    python tools/jetson_preflight.py --skip-models     # fast, no inference
    python tools/jetson_preflight.py --json            # machine-readable
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Make the repo importable when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

_COLORS = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", SKIP: "\033[90m"}
_RESET = "\033[0m"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    action: str = ""
    data: dict = field(default_factory=dict)


class Preflight:
    def __init__(self, skip_models: bool = False, config_path: str = "config/system.yaml"):
        self.skip_models = skip_models
        self.config_path = config_path
        self.results: list[CheckResult] = []
        self._cfg: Optional[dict] = None
        self._use_color = sys.stdout.isatty()

    # -- plumbing -----------------------------------------------------

    def _record(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        color = _COLORS.get(result.status, "") if self._use_color else ""
        reset = _RESET if self._use_color else ""
        print(f"  [{color}{result.status:<4}{reset}] {result.name}")
        if result.detail:
            for line in result.detail.splitlines():
                print(f"         {line}")
        if result.action and result.status in (WARN, FAIL):
            print(f"         -> {result.action}")
        return result

    def _run(self, name: str, fn: Callable[[], CheckResult]) -> CheckResult:
        try:
            return self._record(fn())
        except Exception as exc:  # noqa: BLE001 - a broken check must not stop the rest
            return self._record(CheckResult(
                name=name, status=FAIL,
                detail=f"check itself raised: {type(exc).__name__}: {exc}",
                action="This is a bug in the preflight script, not necessarily in the system.",
            ))

    @staticmethod
    def _shell(cmd: list[str], timeout: float = 5.0) -> Optional[str]:
        if not shutil.which(cmd[0]):
            return None
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return (out.stdout or out.stderr).strip()
        except Exception:  # noqa: BLE001
            return None

    def config(self) -> dict:
        if self._cfg is None:
            from common.config_loader import load_system_config
            self._cfg = load_system_config(self.config_path)
        return self._cfg

    # -- platform -----------------------------------------------------

    def check_platform(self) -> CheckResult:
        model = ""
        for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        model = f.read().decode("utf-8", "ignore").strip("\x00").strip()
                    break
                except OSError:
                    pass

        l4t = ""
        if os.path.exists("/etc/nv_tegra_release"):
            try:
                with open("/etc/nv_tegra_release") as f:
                    l4t = f.readline().strip()
            except OSError:
                pass

        if not model and not l4t:
            return CheckResult(
                "Platform", WARN,
                detail=f"Not a Jetson (python {sys.version.split()[0]}, {sys.platform}).",
                action="This preflight is meant to run ON the Jetson. Results below "
                       "describe THIS machine, not the robot.",
                data={"model": model, "l4t": l4t, "is_jetson": False},
            )

        return CheckResult(
            "Platform", PASS,
            detail=f"{model or 'Tegra device'}\n{l4t}".strip(),
            data={"model": model, "l4t": l4t, "is_jetson": True},
        )

    def check_power_mode(self) -> CheckResult:
        out = self._shell(["nvpmodel", "-q"])
        if out is None:
            return CheckResult("Power mode", SKIP,
                               detail="nvpmodel not available (not a Jetson?).")

        mode_name = ""
        for line in out.splitlines():
            if "NV Power Mode" in line:
                mode_name = line.split(":")[-1].strip()

        clocks_hint = "Also run `sudo jetson_clocks` to pin clocks to max."
        if mode_name and any(t in mode_name.upper() for t in ("15W", "25W", "MAXN")):
            return CheckResult("Power mode", PASS,
                               detail=f"{mode_name}. {clocks_hint}",
                               data={"mode": mode_name})

        return CheckResult(
            "Power mode", WARN,
            detail=f"Current mode: {mode_name or out.splitlines()[0] if out else 'unknown'}",
            action=f"A low-power mode roughly halves inference throughput. "
                   f"`sudo nvpmodel -m 0` for max performance. {clocks_hint}",
            data={"mode": mode_name},
        )

    # -- python stack -----------------------------------------------------

    def check_torch(self) -> CheckResult:
        try:
            import torch
        except ImportError:
            return CheckResult(
                "PyTorch", FAIL,
                detail="torch is not installed.",
                action="Install NVIDIA's Jetson wheel, NOT `pip install torch` "
                       "(PyPI has no CUDA-enabled aarch64 build — you would get a "
                       "CPU-only or wrong-arch wheel). See the Jetson section of the README.",
            )

        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        built_cuda = getattr(torch.version, "cuda", None)

        if not cuda_available:
            return CheckResult(
                "PyTorch", FAIL,
                detail=f"torch {version}, torch.version.cuda={built_cuda}, "
                       f"cuda.is_available()=False",
                action="This is the classic wrong-wheel symptom: a CPU-only torch on the "
                       "Jetson. YOLO and depth will silently fall back to CPU and run "
                       "several times slower. Reinstall NVIDIA's Jetson PyTorch wheel "
                       "matching your JetPack version.",
                data={"version": version, "cuda": False},
            )

        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = "unknown CUDA device"

        return CheckResult(
            "PyTorch", PASS,
            detail=f"torch {version} (CUDA {built_cuda}) on {name}",
            data={"version": version, "cuda": True, "device": name},
        )

    def check_package(self, module: str, label: str, required: bool,
                      install_hint: str) -> CheckResult:
        try:
            mod = __import__(module)
        except ImportError:
            return CheckResult(
                label, FAIL if required else WARN,
                detail=f"{module} is not installed.",
                action=install_hint,
                data={"installed": False},
            )
        version = getattr(mod, "__version__", "unknown")
        return CheckResult(label, PASS, detail=f"{label} {version}",
                           data={"installed": True, "version": version})

    def check_ultralytics(self) -> CheckResult:
        return self.check_package(
            "ultralytics", "Ultralytics YOLO", required=True,
            install_hint="pip install ultralytics --no-deps  (plain `pip install "
                         "ultralytics` pulls its own torch/torchvision and will "
                         "overwrite the Jetson CUDA build). Then install its remaining "
                         "deps by hand — see the README.",
        )

    def check_transformers(self) -> CheckResult:
        result = self.check_package(
            "transformers", "Transformers (depth)", required=False,
            install_hint="pip install transformers  — only needed to run a REAL depth "
                         "model. Without it the pipeline runs the mock depth generator.",
        )
        if result.status == WARN:
            result.detail += "\nDepth will run the MOCK generator (synthetic values)."
        return result

    def check_opencv(self) -> CheckResult:
        try:
            import cv2
        except ImportError:
            return CheckResult("OpenCV", FAIL, detail="cv2 is not installed.",
                               action="pip install opencv-python  (or use the JetPack build)")

        has_v4l2 = hasattr(cv2, "CAP_V4L2")
        detail = f"OpenCV {cv2.__version__}"
        if not has_v4l2:
            return CheckResult("OpenCV", WARN, detail=detail,
                               action="This build lacks CAP_V4L2; camera tuning "
                                      "(pixel format, buffer depth) will not apply.")
        return CheckResult("OpenCV", PASS, detail=detail,
                           data={"version": cv2.__version__})

    # -- camera -----------------------------------------------------

    def check_video_devices(self) -> CheckResult:
        devices = sorted(glob.glob("/dev/video*"))
        if not devices:
            return CheckResult(
                "Video devices", FAIL,
                detail="No /dev/video* devices found.",
                action="Check the USB cable and `dmesg | tail` after plugging the camera in. "
                       "A camera on USB 3 that only enumerates on USB 2 is a common cause.",
                data={"devices": []},
            )

        detail_lines = [f"Found: {', '.join(devices)}"]
        formats: dict[str, str] = {}
        for dev in devices[:4]:
            out = self._shell(["v4l2-ctl", "--list-formats-ext", "-d", dev], timeout=5.0)
            if out:
                fmts = sorted({line.split("'")[1] for line in out.splitlines()
                               if "'" in line and "]" in line})
                if fmts:
                    formats[dev] = ",".join(fmts)
                    detail_lines.append(f"{dev}: {formats[dev]}")

        if not formats and not shutil.which("v4l2-ctl"):
            detail_lines.append("(install v4l-utils for per-device format details)")

        mjpg_somewhere = any("MJPG" in v for v in formats.values())
        if formats and not mjpg_somewhere:
            return CheckResult(
                "Video devices", WARN, detail="\n".join(detail_lines),
                action="No device advertises MJPG. Uncompressed-only cameras are limited "
                       "to roughly 5-10 FPS at 720p over USB 2.0 — either lower the "
                       "resolution in config/camera.yaml or set hardware.fourcc to a "
                       "format this camera supports.",
                data={"devices": devices, "formats": formats},
            )

        return CheckResult("Video devices", PASS, detail="\n".join(detail_lines),
                           data={"devices": devices, "formats": formats})

    def check_camera_capture(self) -> CheckResult:
        from hardware.base import HardwareUnavailableError
        from hardware.camera import DeviceCamera, build_camera_from_config

        cfg = self.config()
        camera = build_camera_from_config(cfg)

        if not isinstance(camera, DeviceCamera):
            return CheckResult(
                "Camera capture", WARN,
                detail=f"Config resolves to {camera.describe()}, not a real device.",
                action="Set camera.source_type: \"device\" (or simulation.kind: \"webcam\") "
                       "in config/camera.yaml to test the real camera.",
                data={"real_device": False},
            )

        try:
            camera.open()
        except HardwareUnavailableError as exc:
            return CheckResult(
                "Camera capture", FAIL, detail=str(exc),
                action="Check the device path/index in config/camera.yaml and that your "
                       "user can read it (`ls -l /dev/video0`; add yourself to the "
                       "`video` group if not).",
                data={"real_device": True, "opened": False},
            )

        try:
            warmup_deadline = time.time() + 1.0
            while time.time() < warmup_deadline:
                camera.read()

            frames = 0
            started = time.time()
            while time.time() - started < 3.0:
                if camera.read() is not None:
                    frames += 1
            measured_fps = frames / max(time.time() - started, 1e-6)
        finally:
            camera.close()

        requested_fps = float(cfg.get("camera", {}).get("fps", 30))
        detail = (f"{camera.width}x{camera.height} fourcc="
                  f"{camera.negotiated_fourcc or 'unknown'} "
                  f"measured {measured_fps:.1f} FPS (requested {requested_fps:.0f})")
        data = {
            "real_device": True, "opened": True, "width": camera.width,
            "height": camera.height, "fourcc": camera.negotiated_fourcc,
            "measured_fps": round(measured_fps, 2),
        }

        if frames == 0:
            return CheckResult("Camera capture", FAIL, detail=detail,
                               action="Camera opened but delivered no frames. Try another "
                                      "USB port, and check `dmesg | tail`.",
                               data=data)

        if measured_fps < requested_fps * 0.6:
            return CheckResult(
                "Camera capture", WARN, detail=detail,
                action="Well below the requested rate. Usual cause is an uncompressed "
                       "pixel format: confirm fourcc is MJPG above, and check "
                       "`v4l2-ctl --list-formats-ext`. Lowering resolution also helps.",
                data=data,
            )

        return CheckResult("Camera capture", PASS, detail=detail, data=data)

    # -- models -----------------------------------------------------

    def check_yolo(self) -> CheckResult:
        if self.skip_models:
            return CheckResult("YOLO inference", SKIP, detail="--skip-models")

        import numpy as np

        from perception.detector.yolo_detector import build_detector_from_config

        cfg = self.config()
        detector = build_detector_from_config(cfg)
        detector.initialize()

        if getattr(detector, "is_degraded", False):
            return CheckResult(
                "YOLO inference", FAIL,
                detail="Detector initialized in degraded mode (model unavailable).",
                action="Check that ultralytics is installed and that the weights at "
                       "perception.detector.model_path exist or can be downloaded.",
                data={"degraded": True},
            )

        frame = np.random.randint(0, 255, size=(720, 1280, 3), dtype=np.uint8)
        detector.infer(frame)                       # warm-up (first call is slow)
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            detector.infer(frame)
            timings.append((time.perf_counter() - started) * 1000.0)
        detector.shutdown()

        median_ms = sorted(timings)[len(timings) // 2]
        info = detector.get_model_info()
        detail = (f"{info.get('model_path')} on {info.get('device')} "
                  f"imgsz={info.get('image_size')}: {median_ms:.0f} ms/frame "
                  f"(~{1000 / max(median_ms, 1e-6):.1f} FPS)")
        data = {"median_ms": round(median_ms, 1), "device": info.get("device")}

        if str(info.get("device", "")).startswith("cpu"):
            return CheckResult(
                "YOLO inference", WARN, detail=detail,
                action="Running on CPU. Fix the PyTorch CUDA check above, or set "
                       "perception.detector.device to a CUDA device.",
                data=data,
            )
        return CheckResult("YOLO inference", PASS, detail=detail, data=data)

    def check_depth(self) -> CheckResult:
        if self.skip_models:
            return CheckResult("Depth inference", SKIP, detail="--skip-models")

        import numpy as np

        from perception.depth.da3_depth_estimator import build_depth_estimator_from_config

        cfg = self.config()
        estimator = build_depth_estimator_from_config(cfg)
        estimator.initialize()
        info = estimator.get_model_info()

        frame = np.random.randint(0, 255, size=(720, 1280, 3), dtype=np.uint8)
        estimator.infer_sync(frame)                 # warm-up
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            estimator.infer_sync(frame)
            timings.append((time.perf_counter() - started) * 1000.0)
        estimator.shutdown()

        median_ms = sorted(timings)[len(timings) // 2]
        detail = (f"{info.get('model')} on {info.get('device')} "
                  f"@{info.get('inference_resolution')}: {median_ms:.0f} ms/frame "
                  f"(~{1000 / max(median_ms, 1e-6):.1f} FPS), "
                  f"kind={info.get('depth_kind')}")
        data = {"median_ms": round(median_ms, 1), "simulated": bool(info.get("simulated"))}

        if info.get("simulated"):
            return CheckResult(
                "Depth inference", WARN,
                detail=detail + "\nThis is the MOCK generator — synthetic values, no real "
                                "scene information.",
                action="Set perception.depth.model_path (and pip install transformers) to "
                       "run a real model. See the README's Jetson section.",
                data=data,
            )

        if str(info.get("device", "")).startswith("cpu"):
            return CheckResult("Depth inference", WARN, detail=detail,
                               action="Depth is running on CPU and will dominate frame time. "
                                      "Fix the PyTorch CUDA check above.",
                               data=data)

        return CheckResult("Depth inference", PASS, detail=detail, data=data)

    # -- lidar (presence only; the runtime issue is separate) -----------------

    def check_lidar_port(self) -> CheckResult:
        candidates = ["/dev/ydlidar"] + sorted(glob.glob("/dev/ttyUSB*"))
        present = [p for p in candidates if os.path.exists(p)]
        if not present:
            return CheckResult(
                "LiDAR port", WARN,
                detail="No /dev/ydlidar or /dev/ttyUSB* found.",
                action="Only matters when you want the LiDAR; perception does not depend "
                       "on it.",
                data={"ports": []},
            )
        readable = [p for p in present if os.access(p, os.R_OK | os.W_OK)]
        detail = f"Present: {', '.join(present)}"
        if not readable:
            return CheckResult("LiDAR port", WARN, detail=detail,
                               action="Port exists but is not read/writable by this user. "
                                      "Add yourself to the `dialout` group and re-login.",
                               data={"ports": present, "readable": []})
        return CheckResult("LiDAR port", PASS,
                           detail=detail + f"\nReadable: {', '.join(readable)}",
                           data={"ports": present, "readable": readable})

    # -- driver -----------------------------------------------------

    def run(self) -> int:
        print("\nSentinelSwarm-3D — Jetson preflight\n" + "=" * 60)

        print("\nPlatform")
        self._run("Platform", self.check_platform)
        self._run("Power mode", self.check_power_mode)

        print("\nPython stack")
        self._run("PyTorch", self.check_torch)
        self._run("Ultralytics YOLO", self.check_ultralytics)
        self._run("Transformers (depth)", self.check_transformers)
        self._run("OpenCV", self.check_opencv)

        print("\nCamera")
        self._run("Video devices", self.check_video_devices)
        self._run("Camera capture", self.check_camera_capture)

        print("\nModels")
        self._run("YOLO inference", self.check_yolo)
        self._run("Depth inference", self.check_depth)

        print("\nLiDAR")
        self._run("LiDAR port", self.check_lidar_port)

        return self.summarize()

    def summarize(self) -> int:
        counts = {s: sum(1 for r in self.results if r.status == s)
                  for s in (PASS, WARN, FAIL, SKIP)}
        print("\n" + "=" * 60)
        print(f"  {counts[PASS]} passed, {counts[WARN]} warnings, "
              f"{counts[FAIL]} failures, {counts[SKIP]} skipped")

        failures = [r for r in self.results if r.status == FAIL]
        if failures:
            print("\n  Blocking issues:")
            for r in failures:
                print(f"    - {r.name}: {r.action or r.detail}")
            print("\n  Result: NOT READY")
            return 1

        warnings = [r for r in self.results if r.status == WARN]
        if warnings:
            print("\n  Non-blocking warnings:")
            for r in warnings:
                print(f"    - {r.name}: {r.action or r.detail}")
            print("\n  Result: READY (with caveats above)")
            return 0

        print("\n  Result: READY")
        return 0

    def as_json(self) -> str:
        return json.dumps(
            {
                "results": [
                    {"name": r.name, "status": r.status, "detail": r.detail,
                     "action": r.action, "data": r.data}
                    for r in self.results
                ],
                "ready": not any(r.status == FAIL for r in self.results),
            },
            indent=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Jetson bring-up preflight for SentinelSwarm-3D")
    parser.add_argument("--config", default="config/system.yaml")
    parser.add_argument("--skip-models", action="store_true",
                        help="Skip YOLO/depth load and timing (much faster)")
    parser.add_argument("--json", action="store_true",
                        help="Also emit machine-readable JSON at the end")
    args = parser.parse_args()

    preflight = Preflight(skip_models=args.skip_models, config_path=args.config)
    exit_code = preflight.run()

    if args.json:
        print("\n--- JSON ---")
        print(preflight.as_json())

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
