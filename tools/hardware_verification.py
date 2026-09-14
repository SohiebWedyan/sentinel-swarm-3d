#!/usr/bin/env python3
"""Phase 2 hardware verification — run this ON the Jetson.

Produces ONE artifact that answers a single question:

    Is Phase 2 HARDWARE VERIFIED, or which blockers remain?

It runs the preflight checks, then exercises the real pipeline under
sustained load, captures an annotated frame as visual evidence, and writes
everything into a self-contained ZIP:

    hardware_verification_<host>_<timestamp>.zip
        report.md          human-readable, with the verdict at the top
        report.json        machine-readable, for diffing between runs
        sample_frame.jpg   annotated detections + depth (evidence)
        run.log            full log output

The verdict criteria are stated explicitly in the report so the result can
be argued with rather than taken on faith. LiDAR is reported but is NOT a
blocker for Phase 2 — the perception pipeline does not depend on it, and
its runtime issue is separately deferred.

Nothing here commands motion.

    python tools/hardware_verification.py
    python tools/hardware_verification.py --seconds 20 --out /tmp
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import platform
import socket
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.jetson_preflight import FAIL, PASS, SKIP, WARN, Preflight  # noqa: E402

VERIFIED = "HARDWARE VERIFIED"
BLOCKED = "BLOCKED"


@dataclass
class SustainedResult:
    """Outcome of running a subsystem under load for a fixed duration."""

    ran: bool = False
    duration_s: float = 0.0
    frames: int = 0
    fps: float = 0.0
    detail: str = ""
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


class HardwareVerification:
    def __init__(self, seconds: float = 15.0, config_path: str = "config/system.yaml"):
        self.seconds = seconds
        self.config_path = config_path
        self.preflight = Preflight(skip_models=False, config_path=config_path)
        self.camera_run = SustainedResult()
        self.perception_run = SustainedResult()
        self.sample_frame_jpg: Optional[bytes] = None
        self.log_stream = io.StringIO()
        self._log_handler: Optional[logging.Handler] = None
        self.configured_mode: str = "unknown"
        self.host_is_jetson: bool = False

    # -- log capture -----------------------------------------------------

    def _start_log_capture(self) -> None:
        handler = logging.StreamHandler(self.log_stream)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        self._log_handler = handler

    def _stop_log_capture(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

    # -- sustained runs -----------------------------------------------------

    def run_camera_sustained(self) -> SustainedResult:
        """Capture continuously for `seconds` and measure what the pipeline
        would really receive — not a single lucky frame.
        """
        from runtime.camera_process import build_camera_process_core

        result = SustainedResult()
        try:
            core = build_camera_process_core(self.preflight.config())
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        core.start()
        try:
            time.sleep(1.0)   # let the camera settle before measuring
            started = time.time()
            consumed = 0
            last_seq = 0
            while time.time() - started < self.seconds:
                packet = core.get_latest_frame()
                if packet is not None:
                    consumed += 1
                    last_seq = packet.sequence_id
                time.sleep(0.005)

            elapsed = time.time() - started
            status = core.get_status()

            result.ran = True
            result.duration_s = round(elapsed, 2)
            result.frames = consumed
            result.fps = round(consumed / max(elapsed, 1e-6), 2)
            result.extra = {
                "device": status.device,
                "simulated": status.simulated,
                "width": status.width,
                "height": status.height,
                "configured_fps": status.configured_fps,
                "camera_measured_fps": status.measured_fps,
                "frames_dropped": status.frames_dropped,
                "reconnect_attempts": status.reconnect_attempts,
                "healthy": status.healthy,
                "last_sequence_id": last_seq,
                "error_message": status.error_message,
            }
            result.detail = (f"{status.device} {status.width}x{status.height} — "
                             f"consumed {consumed} frames in {elapsed:.1f}s "
                             f"({result.fps:.1f} FPS)")
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            core.stop()
        return result

    def run_perception_sustained(self) -> SustainedResult:
        """Drive the real camera through YOLO + depth for `seconds`, in this
        process, and record what the pipeline actually sustains.
        """
        from runtime.camera_process import build_camera_process_core
        from runtime.perception_process import build_perception_process_core

        result = SustainedResult()
        try:
            cfg = self.preflight.config()
            camera = build_camera_process_core(cfg)
            perception = build_perception_process_core(cfg)
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        camera.start()
        try:
            perception.start()
        except Exception as exc:  # noqa: BLE001
            result.error = f"model initialization failed: {type(exc).__name__}: {exc}"
            camera.stop()
            return result

        detector_ms: list[float] = []
        depth_ms: list[float] = []
        latency_ms: list[float] = []
        detection_counts: list[int] = []
        processed = 0
        last_result = None

        try:
            time.sleep(1.0)
            started = time.time()
            while time.time() - started < self.seconds:
                packet = camera.get_latest_frame()
                if packet is None:
                    time.sleep(0.005)
                    continue
                res = perception.process_frame(packet)
                if res is None:
                    continue
                processed += 1
                last_result = res
                detector_ms.append(res.timing.detector_ms)
                if res.depth is not None:
                    depth_ms.append(res.timing.depth_ms)
                latency_ms.append(res.timing.pipeline_latency_ms)
                detection_counts.append(res.detection_count)

            elapsed = time.time() - started
            health = perception.get_health()

            if last_result is not None:
                self.sample_frame_jpg = self._render_sample(cfg, last_result, health,
                                                            camera.get_status())

            def _median(values: list[float]) -> float:
                return round(sorted(values)[len(values) // 2], 2) if values else 0.0

            result.ran = True
            result.duration_s = round(elapsed, 2)
            result.frames = processed
            result.fps = round(processed / max(elapsed, 1e-6), 2)
            result.extra = {
                "detector_ms_median": _median(detector_ms),
                "depth_ms_median": _median(depth_ms),
                "latency_ms_median": _median(latency_ms),
                "latency_ms_max": round(max(latency_ms), 2) if latency_ms else 0.0,
                "frames_with_depth": len(depth_ms),
                "total_detections": sum(detection_counts),
                "max_detections_in_a_frame": max(detection_counts) if detection_counts else 0,
                "detector_degraded": health.detector_degraded,
                "depth_simulated": health.depth_simulated,
                "depth_kind": health.depth_kind.value,
                "depth_model": health.depth_model,
                "detector_model": health.detector_model,
                "device": health.device,
                "frames_dropped_stale": health.frames_dropped_stale,
                "healthy": health.healthy,
                "error_message": health.error_message,
            }
            result.detail = (f"{processed} frames in {elapsed:.1f}s ({result.fps:.1f} FPS) — "
                             f"yolo {result.extra['detector_ms_median']:.0f}ms, "
                             f"depth {result.extra['depth_ms_median']:.0f}ms, "
                             f"latency {result.extra['latency_ms_median']:.0f}ms")
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            perception.stop()
            camera.stop()
        return result

    def _render_sample(self, cfg, result, health, camera_status) -> Optional[bytes]:
        """Annotated frame as evidence that detections/depth are real."""
        try:
            import cv2

            from visualization.perception_view import PerceptionView

            view = PerceptionView(show_depth=True)
            view._gui_available = False           # never try to open a window here
            canvas = view.render(result, health, camera_status)
            if canvas is None:
                return None
            ok, buf = cv2.imencode(".jpg", canvas)
            return buf.tobytes() if ok else None
        except Exception:  # noqa: BLE001 - evidence is nice to have, not essential
            return None

    # -- verdict -----------------------------------------------------

    def compute_verdict(self) -> tuple[str, list[str], list[str]]:
        """Explicit, arguable criteria. Returns (verdict, blockers, warnings).

        Scope note: LiDAR is deliberately excluded. Phase 2's perception
        pipeline does not depend on it, and its runtime issue is tracked
        separately — letting it block this verdict would conflate two
        independent things.
        """
        blockers: list[str] = []
        warnings: list[str] = []

        by_name = {r.name: r for r in self.preflight.results}

        # 0. Mode. Verification is a claim about REAL hardware, so running it
        # while the system is configured for simulation is reported as a
        # blocker in its own right — otherwise a bundle full of synthetic
        # measurements could be mistaken for evidence.
        if self.configured_mode != "live":
            blockers.append(
                f"system.mode is \"{self.configured_mode}\", not \"live\". Hardware "
                f"verification must run against real hardware — set system.mode: "
                f"\"live\" in config/system.yaml and re-run on the robot."
            )

        def status_of(name: str) -> str:
            return by_name[name].status if name in by_name else SKIP

        # 1. CUDA-capable PyTorch
        if status_of("PyTorch") == FAIL:
            blockers.append("PyTorch is missing or has no CUDA — inference would run on CPU. "
                            "Install NVIDIA's Jetson wheel.")

        # 2. YOLO real and on GPU
        if status_of("Ultralytics YOLO") == FAIL:
            blockers.append("Ultralytics is not installed — no detections are possible.")
        if status_of("YOLO inference") == FAIL:
            blockers.append("YOLO failed to load or run.")
        elif status_of("YOLO inference") == WARN:
            warnings.append("YOLO is running on CPU rather than the GPU.")

        # 3. Camera is a real device delivering frames at a usable rate
        cam = self.camera_run
        if not cam.ran:
            blockers.append(f"Camera capture did not run: {cam.error or 'unknown error'}")
        else:
            if cam.extra.get("simulated"):
                blockers.append("Camera is SYNTHETIC, not a real device — set "
                                "camera.source_type: \"device\" in config/camera.yaml.")
            if cam.frames == 0:
                blockers.append("Camera delivered zero frames.")
            else:
                configured = float(cam.extra.get("configured_fps") or 0)
                if configured and cam.fps < configured * 0.5:
                    warnings.append(
                        f"Camera sustained {cam.fps:.1f} FPS against a configured "
                        f"{configured:.0f} — check the negotiated pixel format (MJPG vs YUYV)."
                    )

        # 4. The pipeline sustains real work
        per = self.perception_run
        if not per.ran:
            blockers.append(f"Perception did not run: {per.error or 'unknown error'}")
        else:
            if per.extra.get("detector_degraded"):
                blockers.append("Detector is degraded (no model) — zero detections.")
            if per.frames == 0:
                blockers.append("Perception processed zero frames.")
            if per.extra.get("depth_simulated"):
                blockers.append("Depth is the MOCK generator, not a real model — set "
                                "perception.depth.model_path and install transformers.")
            if per.extra.get("total_detections", 0) == 0 and not per.extra.get("detector_degraded"):
                warnings.append("Real detector ran but found nothing in the whole run — "
                                "point the camera at something recognizable (a person, a "
                                "chair) and re-run before trusting this.")
            latency = per.extra.get("latency_ms_median", 0)
            if latency and latency > 500:
                warnings.append(f"Median end-to-end latency {latency:.0f} ms is high; "
                                f"consider depth_every_n_frames or a smaller imgsz.")

        # 5. Reported, never blocking
        if status_of("LiDAR port") in (WARN, FAIL):
            warnings.append("LiDAR port not usable (reported only — not a Phase 2 blocker).")

        verdict = VERIFIED if not blockers else BLOCKED
        return verdict, blockers, warnings

    # -- report -----------------------------------------------------

    def build_report(self) -> tuple[str, dict]:
        verdict, blockers, warnings = self.compute_verdict()
        now = datetime.now(timezone.utc)

        payload: dict[str, Any] = {
            "verdict": verdict,
            "blockers": blockers,
            "warnings": warnings,
            "generated_utc": now.isoformat(),
            "host": socket.gethostname(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "config_path": self.config_path,
            "configured_mode": self.configured_mode,
            "host_is_jetson": self.host_is_jetson,
            "measurement_seconds": self.seconds,
            "preflight": [
                {"name": r.name, "status": r.status, "detail": r.detail,
                 "action": r.action, "data": r.data}
                for r in self.preflight.results
            ],
            "camera_sustained": asdict(self.camera_run),
            "perception_sustained": asdict(self.perception_run),
        }

        lines: list[str] = []
        lines.append("# SentinelSwarm-3D — Phase 2 Hardware Verification")
        lines.append("")
        lines.append(f"## VERDICT: {verdict}")
        lines.append("")
        if blockers:
            lines.append(f"**{len(blockers)} blocker(s):**")
            lines.append("")
            for b in blockers:
                lines.append(f"- {b}")
            lines.append("")
        else:
            lines.append("All Phase 2 criteria met on real hardware.")
            lines.append("")
        if warnings:
            lines.append("**Warnings (non-blocking):**")
            lines.append("")
            for w in warnings:
                lines.append(f"- {w}")
            lines.append("")

        lines.append("## Criteria used")
        lines.append("")
        lines.append("A run is HARDWARE VERIFIED when all of the following hold:")
        lines.append("")
        lines.append("1. PyTorch is present with CUDA available.")
        lines.append("2. Ultralytics is installed and YOLO loads and runs.")
        lines.append("3. The camera is a REAL device (not synthetic) and delivers frames.")
        lines.append("4. Perception processes frames with a non-degraded detector.")
        lines.append("5. Depth is a REAL model, not the mock generator.")
        lines.append("")
        lines.append("LiDAR is reported but excluded: Phase 2 perception does not depend "
                     "on it, and its runtime issue is tracked separately.")
        lines.append("")

        lines.append("## Environment")
        lines.append("")
        lines.append(f"- Host: `{payload['host']}`")
        lines.append(f"- Jetson: **{'yes' if self.host_is_jetson else 'NO'}**")
        lines.append(f"- system.mode: **{self.configured_mode}**")
        lines.append(f"- Platform: `{payload['platform']}` ({payload['machine']})")
        lines.append(f"- Python: `{payload['python']}`")
        lines.append(f"- Generated: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("")

        lines.append("## Preflight checks")
        lines.append("")
        lines.append("| Check | Status | Detail |")
        lines.append("|---|---|---|")
        for r in self.preflight.results:
            detail = (r.detail or "").replace("\n", "; ").replace("|", "\\|")
            lines.append(f"| {r.name} | {r.status} | {detail} |")
        lines.append("")

        lines.append(f"## Sustained camera ({self.seconds:.0f}s)")
        lines.append("")
        if self.camera_run.ran:
            lines.append(f"{self.camera_run.detail}")
            lines.append("")
            for k, v in self.camera_run.extra.items():
                lines.append(f"- `{k}`: {v}")
        else:
            lines.append(f"Did not run: {self.camera_run.error}")
        lines.append("")

        lines.append(f"## Sustained perception ({self.seconds:.0f}s)")
        lines.append("")
        if self.perception_run.ran:
            lines.append(f"{self.perception_run.detail}")
            lines.append("")
            for k, v in self.perception_run.extra.items():
                lines.append(f"- `{k}`: {v}")
        else:
            lines.append(f"Did not run: {self.perception_run.error}")
        lines.append("")

        if self.sample_frame_jpg:
            lines.append("## Evidence")
            lines.append("")
            lines.append("`sample_frame.jpg` in this bundle is an annotated frame from the "
                         "run (detections + depth panel).")
            lines.append("")

        lines.append("## Reminder on depth units")
        lines.append("")
        lines.append("Monocular depth output is RELATIVE — unitless and up to an unknown "
                     "scale. A verified run does NOT mean metric distance is available.")
        lines.append("")

        return "\n".join(lines), payload

    def write_bundle(self, out_dir: str) -> str:
        report_md, payload = self.build_report()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        host = socket.gethostname().replace(" ", "_")
        os.makedirs(out_dir, exist_ok=True)
        zip_path = os.path.join(out_dir, f"hardware_verification_{host}_{stamp}.zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("report.md", report_md)
            zf.writestr("report.json", json.dumps(payload, indent=2, default=str))
            zf.writestr("run.log", self.log_stream.getvalue())
            if self.sample_frame_jpg:
                zf.writestr("sample_frame.jpg", self.sample_frame_jpg)

        return zip_path

    # -- driver -----------------------------------------------------

    def run(self, out_dir: str) -> int:
        self._start_log_capture()
        try:
            print("\nSentinelSwarm-3D — Phase 2 hardware verification")
            print("=" * 60)
            try:
                self.configured_mode = str(
                    self.preflight.config().get("system", {}).get("mode", "unknown")
                ).lower()
            except Exception:  # noqa: BLE001
                self.configured_mode = "unknown"
            print(f"  system.mode = {self.configured_mode}")

            self.preflight.run()
            platform_result = next(
                (r for r in self.preflight.results if r.name == "Platform"), None
            )
            if platform_result is not None:
                self.host_is_jetson = bool(platform_result.data.get("is_jetson", False))

            print(f"\nSustained camera ({self.seconds:.0f}s)")
            self.camera_run = self.run_camera_sustained()
            print(f"  {self.camera_run.detail or self.camera_run.error}")

            print(f"\nSustained perception ({self.seconds:.0f}s)")
            self.perception_run = self.run_perception_sustained()
            print(f"  {self.perception_run.detail or self.perception_run.error}")
        finally:
            self._stop_log_capture()

        verdict, blockers, warnings = self.compute_verdict()
        zip_path = self.write_bundle(out_dir)

        print("\n" + "=" * 60)
        print(f"  VERDICT: {verdict}")
        if blockers:
            print(f"\n  {len(blockers)} blocker(s):")
            for b in blockers:
                print(f"    - {b}")
        if warnings:
            print(f"\n  {len(warnings)} warning(s):")
            for w in warnings:
                print(f"    - {w}")
        print(f"\n  Report bundle: {zip_path}")
        print("  Send this ZIP back for comparison.\n")

        return 0 if verdict == VERIFIED else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2 hardware verification report (run on the Jetson)"
    )
    parser.add_argument("--config", default="config/system.yaml")
    parser.add_argument("--seconds", type=float, default=15.0,
                        help="Duration of each sustained measurement (default 15)")
    parser.add_argument("--out", default=".", help="Directory for the report bundle")
    args = parser.parse_args()

    from common.logging_utils import configure_logging

    configure_logging(level="INFO", robot_id="leader")

    verification = HardwareVerification(seconds=args.seconds, config_path=args.config)
    return verification.run(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
