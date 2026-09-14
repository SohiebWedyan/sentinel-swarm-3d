"""Intel RealSense D435-only sensor validation.

This module deliberately does *not* publish motion commands, SLAM inputs, or
IMU data.  A D435 has no built-in IMU (unlike a D435i), so validation is
limited to its colour/stereo-depth pipeline: RGB, depth, alignment, point
cloud, USB link and delivered frame rate.

``pyrealsense2`` is imported only when a real session is opened.  The small
session seam makes all validation logic testable in CI without a camera.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"


@dataclass(frozen=True)
class D435Frame:
    """One colour/depth pair after depth has been aligned to colour."""

    timestamp_s: float
    color: np.ndarray
    depth: np.ndarray
    points_xyz: np.ndarray


@dataclass(frozen=True)
class _FrameObservation:
    """Small, detached validation data for one frame.

    Do not retain numpy views returned by librealsense: their base objects
    keep SDK frames alive, eventually exhausting the RSUSB frame queue.
    """

    timestamp_s: float
    color_shape: tuple[int, ...]
    depth_shape: tuple[int, ...]
    valid_depth_pixels: int
    finite_points: int
    nonzero_points: int


class D435Session(Protocol):
    def open(self) -> dict[str, Any]: ...
    def next_frame(self, timeout_ms: int) -> D435Frame: ...
    def close(self) -> None: ...


@dataclass
class Check:
    name: str
    status: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class D435ValidationReport:
    device: dict[str, Any]
    checks: list[Check]
    rgb_fps: float = 0.0
    depth_fps: float = 0.0
    sample_count: int = 0

    @property
    def overall_status(self) -> str:
        return FAIL if any(c.status == FAIL for c in self.checks) else PASS

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "overall_status": self.overall_status}


class PyRealSenseD435Session:
    """Thin adapter around librealsense, kept out of import-time code."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.rs: Any = None
        self.pipeline: Any = None
        self.align: Any = None
        self.pointcloud: Any = None

    def open(self) -> dict[str, Any]:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed. Install librealsense Python bindings on the Jetson."
            ) from exc
        self.rs = rs
        context = rs.context()
        devices = context.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No Intel RealSense device detected")
        device = devices[0]
        name = device.get_info(rs.camera_info.name)
        if "D435" not in name.upper() or "D435I" in name.upper():
            raise RuntimeError(f"Expected Intel RealSense D435 (not D435i); detected {name!r}")

        # Use librealsense's default context for the stream pipeline.  We
        # keep the explicit context above for discovery, but the default
        # pipeline is the same path used by the official samples and avoids
        # intermittent stalls observed with an explicitly supplied context on
        # Jetson RSUSB builds.
        self.pipeline = rs.pipeline()
        config = rs.config()
        serial = self.cfg.get("serial_no")
        if serial:
            config.enable_device(str(serial))
        width, height, fps = (int(self.cfg[k]) for k in ("width", "height", "fps"))
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        started_device = profile.get_device()
        self.align = rs.align(rs.stream.color)
        self.pointcloud = rs.pointcloud()

        def info(key: Any) -> str:
            return started_device.get_info(key) if started_device.supports(key) else "unknown"

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_intr = color_profile.get_intrinsics()
        depth_intr = depth_profile.get_intrinsics()
        return {
            "name": info(rs.camera_info.name),
            "serial": info(rs.camera_info.serial_number),
            "firmware": info(rs.camera_info.firmware_version),
            "usb_type": info(rs.camera_info.usb_type_descriptor),
            "color_intrinsics": {"width": color_intr.width, "height": color_intr.height,
                                 "fx": color_intr.fx, "fy": color_intr.fy,
                                 "ppx": color_intr.ppx, "ppy": color_intr.ppy},
            "depth_intrinsics": {"width": depth_intr.width, "height": depth_intr.height,
                                 "fx": depth_intr.fx, "fy": depth_intr.fy,
                                 "ppx": depth_intr.ppx, "ppy": depth_intr.ppy},
        }

    def next_frame(self, timeout_ms: int) -> D435Frame:
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms))
        color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense returned an incomplete colour/depth frameset")
        self.pointcloud.map_to(color_frame)
        points = self.pointcloud.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        return D435Frame(
            timestamp_s=time.monotonic(), color=np.asanyarray(color_frame.get_data()),
            depth=np.asanyarray(depth_frame.get_data()), points_xyz=vertices,
        )

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


class RealSenseD435Validator:
    """Collect a short, bounded capture and assess D435 health."""

    def __init__(self, cfg: dict[str, Any], session_factory: Callable[[dict[str, Any]], D435Session] = PyRealSenseD435Session,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.session_factory = session_factory
        self.clock = clock

    def run(self) -> D435ValidationReport:
        session = self.session_factory(self.cfg)
        checks: list[Check] = []
        metadata: dict[str, Any] = {}
        frames: list[_FrameObservation] = []
        try:
            metadata = session.open()
            name = str(metadata.get("name", "unknown"))
            is_d435 = "D435" in name.upper() and "D435I" not in name.upper()
            checks.append(Check("Device model", PASS if is_d435 else FAIL,
                                f"Detected {name}; expected Intel RealSense D435 (not D435i)"))
            checks.append(self._usb_check(metadata))
            # A D435 can expose streams before automatic exposure/USB traffic
            # has settled.  Do not count these startup frames or treat that
            # brief transition as evidence of a broken camera.
            for _ in range(int(self.cfg.get("warmup_frames", 10))):
                session.next_frame(int(self.cfg.get("frame_timeout_ms", 2000)))
            started = self.clock()
            duration = float(self.cfg.get("sample_duration_s", 3.0))
            minimum = int(self.cfg.get("minimum_frames", 30))
            timeout_ms = int(self.cfg.get("frame_timeout_ms", 2000))
            max_timeouts = int(self.cfg.get("max_consecutive_timeouts", 3))
            consecutive_timeouts = 0
            while self.clock() - started < duration or len(frames) < minimum:
                try:
                    frames.append(self._observe(session.next_frame(timeout_ms)))
                    consecutive_timeouts = 0
                except RuntimeError as exc:
                    if "didn't arrive" not in str(exc).lower():
                        raise
                    consecutive_timeouts += 1
                    logger.warning("D435 frame timeout (%d/%d): %s",
                                   consecutive_timeouts, max_timeouts, exc)
                    if consecutive_timeouts >= max_timeouts:
                        raise RuntimeError(
                            f"D435 did not recover after {max_timeouts} consecutive frame timeouts"
                        ) from exc
                    continue
                # A broken mock clock must not make CI loop indefinitely.
                if len(frames) >= max(minimum * 4, 500):
                    break
            checks.extend(self._frame_checks(frames, metadata))
        except Exception as exc:  # Hardware is expected to fail loudly, but report it cleanly.
            logger.exception("D435 validation failed")
            checks.append(Check("Capture", FAIL, f"{type(exc).__name__}: {exc}"))
        finally:
            try:
                session.close()
            except Exception:  # pragma: no cover - defensive hardware cleanup
                logger.exception("Could not stop RealSense pipeline")

        rgb_fps = depth_fps = self._fps(frames)
        report = D435ValidationReport(metadata, checks, rgb_fps, depth_fps, len(frames))
        logger.info("D435 validation complete: %s (%d frames, RGB %.2f FPS, depth %.2f FPS)",
                    report.overall_status, report.sample_count, report.rgb_fps, report.depth_fps)
        return report

    def _usb_check(self, metadata: dict[str, Any]) -> Check:
        usb = str(metadata.get("usb_type", "unknown"))
        required = str(self.cfg.get("required_usb_prefix", "3."))
        return Check("USB3", PASS if usb.startswith(required) else FAIL,
                     f"Reported USB link: {usb}", {"usb_type": usb, "required_prefix": required})

    @staticmethod
    def _observe(frame: D435Frame) -> _FrameObservation:
        """Extract scalar facts and immediately release SDK-backed arrays."""
        points = frame.points_xyz
        finite = int(np.count_nonzero(np.isfinite(points).all(axis=1))) if points.ndim == 2 else 0
        nonzero = int(np.count_nonzero(np.linalg.norm(points, axis=1) > 0)) if points.ndim == 2 else 0
        return _FrameObservation(
            timestamp_s=frame.timestamp_s, color_shape=tuple(frame.color.shape),
            depth_shape=tuple(frame.depth.shape), valid_depth_pixels=int(np.count_nonzero(frame.depth)),
            finite_points=finite, nonzero_points=nonzero,
        )

    def _frame_checks(self, frames: list[_FrameObservation], metadata: dict[str, Any]) -> list[Check]:
        if not frames:
            return [Check("RGB", FAIL, "No frames received"), Check("Depth", FAIL, "No frames received")]
        first = frames[0]
        color_ok = len(first.color_shape) == 3 and first.color_shape[2] == 3 and np.prod(first.color_shape) > 0
        valid_depth = [f.valid_depth_pixels for f in frames]
        depth_ok = len(first.depth_shape) == 2 and all(n > 0 for n in valid_depth)
        aligned = color_ok and depth_ok and first.color_shape[:2] == first.depth_shape[:2]
        target_fps = float(self.cfg.get("fps", 30))
        delivered = self._fps(frames)
        min_ratio = float(self.cfg.get("minimum_fps_ratio", 0.85))
        intrinsics = metadata.get("color_intrinsics", {})
        intrinsics_ok = all(float(intrinsics.get(k, 0)) > 0 for k in ("fx", "fy"))
        return [
            Check("RGB", PASS if color_ok else FAIL, f"shape={first.color_shape}"),
            Check("Depth", PASS if depth_ok else FAIL,
                  f"shape={first.depth_shape}; valid pixels/frame min={min(valid_depth)}"),
            Check("RGB-Depth alignment", PASS if aligned else FAIL,
                  f"colour={first.color_shape[:2]}, aligned depth={first.depth_shape[:2]}"),
            Check("PointCloud", PASS if first.nonzero_points > 0 else FAIL,
                  f"finite vertices={first.finite_points}, nonzero vertices={first.nonzero_points}"),
            Check("Intrinsics", PASS if intrinsics_ok else FAIL,
                  "Colour intrinsics available" if intrinsics_ok else "Missing/invalid colour intrinsics"),
            Check("FPS", PASS if delivered >= target_fps * min_ratio else WARN,
                  f"RGB/depth {delivered:.2f} FPS; target {target_fps:.1f} FPS",
                  {"delivered_fps": delivered, "target_fps": target_fps}),
        ]

    @staticmethod
    def _fps(frames: list[D435Frame]) -> float:
        if len(frames) < 2:
            return 0.0
        elapsed = frames[-1].timestamp_s - frames[0].timestamp_s
        return round((len(frames) - 1) / elapsed, 2) if elapsed > 0 else 0.0
