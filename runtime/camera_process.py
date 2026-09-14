"""Camera process (Phase 2).

Isolated from perception so a camera stall, USB re-enumeration, or driver
hiccup can't take down inference, and so ProcessManager can restart it
independently.

    USB RGB camera
          |
    capture thread   (blocking VideoCapture.read(), never on the main loop)
          |
    latest frame     (single slot, oldest dropped)
          |
    FramePacket -> LatestValueQueue -> perception_process

One internal thread, matching the LiDAR process's minimal-thread rule. It
exists because OpenCV's read() blocks: keeping it off the publishing loop
is what lets the process still emit status while the camera is wedged.

REAL-TIME RULE: there is exactly one frame slot, in-process and across the
process boundary. A frame that arrives while the previous one is still
unconsumed REPLACES it. Frames are dropped, never queued — latency matters
more than completeness for a robot, and an unbounded queue turns a slow
consumer into unbounded memory growth plus ever-staler data.

Run standalone for camera bring-up:
    python -m runtime.camera_process --seconds 10
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from common.config_loader import get, load_system_config
from common.logging_utils import configure_logging
from common.types import CameraStatus, FramePacket
from hardware.base import HardwareUnavailableError
from hardware.camera import BaseCamera, build_camera_from_config
from runtime.shared_state import LatestValueQueue

logger = logging.getLogger(__name__)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


class CameraProcessCore:
    def __init__(
        self,
        camera: BaseCamera,
        frame_id: str = "camera",
        configured_fps: float = 30.0,
        ipc_scale: float = 1.0,
        reconnect_enabled: bool = True,
        reconnect_initial_delay_s: float = 1.0,
        reconnect_max_delay_s: float = 10.0,
        reconnect_max_attempts: int = 0,   # 0 = unlimited
    ) -> None:
        self.camera = camera
        self.frame_id = frame_id
        self.configured_fps = configured_fps
        self.ipc_scale = ipc_scale if 0.0 < ipc_scale <= 1.0 else 1.0
        self.reconnect_enabled = reconnect_enabled
        self.reconnect_initial_delay_s = reconnect_initial_delay_s
        self.reconnect_max_delay_s = reconnect_max_delay_s
        self.reconnect_max_attempts = reconnect_max_attempts

        self._latest: Optional[FramePacket] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        self._connected = False
        self._streaming = False
        self._sequence_id = 0
        self._last_frame_time: float = 0.0
        self._measured_fps: float = 0.0
        self._frames_dropped = 0
        self._reconnect_attempts = 0
        self._error_message: Optional[str] = None
        self._last_logged_error: Optional[str] = None

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Open the camera and start capturing.

        Never raises on an absent camera: the failure is recorded in
        get_status() and the capture thread keeps retrying (if reconnect is
        enabled), so the runtime stays up and can report the problem.
        """
        self._open_camera()

        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="camera_capture_thread"
        )
        self._capture_thread.start()

    def _open_camera(self) -> bool:
        try:
            self.camera.open()
            self._connected = True
            self._error_message = None
            self._last_logged_error = None
            logger.info("Camera opened: %s", self.camera.describe())
            return True
        except HardwareUnavailableError as exc:
            self._fail(f"camera unavailable: {exc}")
            return False
        except Exception as exc:  # noqa: BLE001 - a driver bug must not kill the process
            self._fail(f"unexpected error opening camera: {type(exc).__name__}: {exc}")
            return False

    def _fail(self, message: str) -> None:
        """Record a failure once at ERROR; repeats drop to DEBUG so a
        persistently missing camera doesn't flood the log.
        """
        repeated = message == self._last_logged_error
        self._error_message = message
        self._connected = False
        self._streaming = False
        (logger.debug if repeated else logger.error)(
            "Camera error | %s%s", message, " (repeated)" if repeated else ""
        )
        self._last_logged_error = message

    # -- capture -----------------------------------------------------

    def _capture_loop(self) -> None:
        backoff = self.reconnect_initial_delay_s
        # A real VideoCapture.read() blocks at the sensor rate, so it paces
        # itself. A synthetic source or a video file returns instantly and
        # would spin this thread at thousands of FPS — burning a core and
        # producing frames nobody can consume. Pace to the configured rate so
        # both kinds of source behave the same way; for a real camera the
        # remaining sleep is ~0 and this costs nothing.
        target_period_s = 1.0 / max(self.configured_fps, 1.0)

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            if not self._connected:
                if not self.reconnect_enabled:
                    self._stop_event.wait(0.25)
                    continue
                if self.reconnect_max_attempts and \
                        self._reconnect_attempts >= self.reconnect_max_attempts:
                    self._stop_event.wait(0.5)
                    continue

                self._reconnect_attempts += 1
                logger.debug("Camera reconnect attempt %d in %.1fs",
                             self._reconnect_attempts, backoff)
                if self._stop_event.wait(backoff):
                    break
                if self._open_camera():
                    logger.info("Camera reconnected after %d attempt(s)",
                                self._reconnect_attempts)
                    backoff = self.reconnect_initial_delay_s
                else:
                    backoff = min(backoff * 2.0, self.reconnect_max_delay_s)
                continue

            try:
                frame = self.camera.read()
            except HardwareUnavailableError as exc:
                self._close_camera()
                self._fail(f"read failed, camera lost: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                self._close_camera()
                self._fail(f"unexpected read error: {type(exc).__name__}: {exc}")
                continue

            if frame is None:
                # A single dropped grab is normal; a camera that has actually
                # gone away shows up as is_healthy() going false.
                if not self.camera.is_healthy():
                    self._close_camera()
                    self._fail("camera stopped reporting healthy after a failed read")
                else:
                    self._stop_event.wait(0.01)
                continue

            self._publish(frame)

            remaining = target_period_s - (time.perf_counter() - loop_start)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _publish(self, frame) -> None:
        image = frame.image
        scale = 1.0

        if self.ipc_scale != 1.0 and _HAS_CV2 and image is not None:
            new_w = max(1, int(image.shape[1] * self.ipc_scale))
            new_h = max(1, int(image.shape[0] * self.ipc_scale))
            image = cv2.resize(image, (new_w, new_h))
            scale = self.ipc_scale

        now = time.time()
        self._sequence_id += 1
        if self._last_frame_time > 0.0:
            delta = now - self._last_frame_time
            if delta > 0:
                instantaneous = 1.0 / delta
                self._measured_fps = (
                    instantaneous if self._measured_fps == 0.0
                    else 0.8 * self._measured_fps + 0.2 * instantaneous
                )
        self._last_frame_time = now
        self._streaming = True
        self._error_message = None

        packet = FramePacket(
            image=image,
            timestamp=frame.timestamp,
            sequence_id=self._sequence_id,
            frame_id=self.frame_id,
            source=self.camera.describe(),
            width=int(image.shape[1]) if image is not None else 0,
            height=int(image.shape[0]) if image is not None else 0,
            capture_fps=self._measured_fps,
            scale_from_capture=scale,
        )

        with self._lock:
            if self._latest is not None:
                # The previous frame was never consumed — count it so a
                # consumer that can't keep up is visible in diagnostics
                # rather than silently invisible.
                self._frames_dropped += 1
            self._latest = packet

    def _close_camera(self) -> None:
        try:
            self.camera.close()
        except Exception:  # noqa: BLE001
            pass
        self._connected = False
        self._streaming = False

    # -- consumption -----------------------------------------------------

    def get_latest_frame(self) -> Optional[FramePacket]:
        """Take the newest frame, clearing the slot. Returns None when no
        NEW frame has arrived since the last call — the caller should not
        reprocess a frame it already handled.
        """
        with self._lock:
            packet, self._latest = self._latest, None
            return packet

    def peek_latest_frame(self) -> Optional[FramePacket]:
        """Look at the newest frame without consuming it (diagnostics)."""
        with self._lock:
            return self._latest

    def is_healthy(self) -> bool:
        if not self._connected or not self._streaming:
            return False
        if self._last_frame_time == 0.0:
            return False
        # Two expected frame intervals of silence (min 1s) means trouble.
        max_gap_s = max(2.0 / max(self.configured_fps, 1.0), 1.0)
        return (time.time() - self._last_frame_time) <= max_gap_s

    def get_status(self) -> CameraStatus:
        width = height = 0
        with self._lock:
            if self._latest is not None:
                width, height = self._latest.width, self._latest.height
        if not width:
            width = getattr(self.camera, "width", 0) or 0
            height = getattr(self.camera, "height", 0) or 0

        return CameraStatus(
            connected=self._connected,
            streaming=self._streaming,
            healthy=self.is_healthy(),
            device=self.camera.describe(),
            width=width,
            height=height,
            configured_fps=self.configured_fps,
            measured_fps=round(self._measured_fps, 2),
            last_frame_timestamp=self._last_frame_time or None,
            sequence_id=self._sequence_id,
            frames_dropped=self._frames_dropped,
            reconnect_attempts=self._reconnect_attempts,
            error_message=self._error_message,
            simulated=bool(getattr(self.camera, "is_simulated", False)),
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        self._close_camera()


def build_camera_process_core(cfg: dict) -> CameraProcessCore:
    cam_cfg = cfg.get("camera", {})
    reconnect_cfg = cam_cfg.get("reconnect", {})
    ipc_cfg = cam_cfg.get("ipc", {})

    return CameraProcessCore(
        camera=build_camera_from_config(cfg),
        frame_id=cam_cfg.get("frame_id", "camera"),
        configured_fps=float(cam_cfg.get("fps", 30)),
        ipc_scale=float(ipc_cfg.get("scale", 1.0)),
        reconnect_enabled=reconnect_cfg.get("enabled", True),
        reconnect_initial_delay_s=float(reconnect_cfg.get("initial_delay_s", 1.0)),
        reconnect_max_delay_s=float(reconnect_cfg.get("max_delay_s", 10.0)),
        reconnect_max_attempts=int(reconnect_cfg.get("max_attempts", 0)),
    )


def run_camera_process_entrypoint(
    cfg: dict,
    frame_output: LatestValueQueue,
    status_output: LatestValueQueue,
    stop_event,
    publish_period_s: float = 0.005,
) -> None:
    """multiprocessing.Process target.

    Publishes FramePacket to frame_output (single-consumer: the perception
    process) and CameraStatus to status_output (single-consumer: the leader
    runtime). Two queues rather than one because LatestValueQueue reads are
    destructive — two consumers sharing one queue would steal frames from
    each other.

    Signal disposition is reset for the same reason as lidar_process: on
    fork this inherits run_leader.py's handler, and terminate() must kill
    this process cleanly rather than raising the parent's exception here.
    """
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    configure_logging(
        level=get(cfg, "system.log_level", "INFO"),
        robot_id=get(cfg, "robots.this_robot", "leader"),
        log_dir=get(cfg, "system.log_dir"),
    )

    health_check_interval_s = float(get(cfg, "camera.health_check_interval", 1.0))

    try:
        core = build_camera_process_core(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.error("camera_process could not be built: %s", exc)
        status_output.put_latest_sync(CameraStatus(error_message=str(exc)))
        return

    logger.info("camera_process starting (source=%s)", core.camera.describe())
    core.start()

    last_status_publish = 0.0
    last_logged_health: Optional[bool] = None
    process_start = time.time()
    startup_grace_s = max(2.0 / max(core.configured_fps, 1.0), 1.0)

    try:
        while not stop_event.is_set():
            packet = core.get_latest_frame()
            if packet is not None:
                frame_output.put_latest(packet)

            now = time.time()
            if (now - last_status_publish) >= health_check_interval_s:
                status = core.get_status()
                status_output.put_latest(status)
                last_status_publish = now

                still_starting = (status.last_frame_timestamp is None
                                  and (now - process_start) < startup_grace_s)
                if (last_logged_health is None or status.healthy != last_logged_health) \
                        and not still_starting:
                    if status.healthy:
                        logger.info("Camera healthy | %s %dx%d @ %.1f FPS",
                                    status.device, status.width, status.height,
                                    status.measured_fps)
                    else:
                        logger.warning(
                            "Camera unhealthy | %s | %s", status.device,
                            status.error_message or "no frames within the expected interval",
                        )
                    last_logged_health = status.healthy

            time.sleep(publish_period_s)
    finally:
        core.stop()
        logger.info("camera_process stopped")


# ---------------------------------------------------------------------------
# Camera-only bring-up diagnostic. No perception, no LiDAR, no motors.
# ---------------------------------------------------------------------------


def run_diagnostic(cfg: dict, seconds: Optional[float] = None, print_period_s: float = 1.0) -> int:
    core = build_camera_process_core(cfg)
    core.start()

    start_time = time.time()
    exit_code = 0
    try:
        while True:
            time.sleep(print_period_s)
            status = core.get_status()
            packet = core.peek_latest_frame()

            print(
                f"[{time.strftime('%H:%M:%S')}] device={status.device} "
                f"connected={status.connected} streaming={status.streaming} "
                f"healthy={status.healthy} {status.width}x{status.height} "
                f"fps={status.measured_fps:.1f}/{status.configured_fps:.0f} "
                f"seq={status.sequence_id} dropped={status.frames_dropped} "
                f"simulated={status.simulated}"
            )
            if status.error_message:
                print(f"    error: {status.error_message}")
            if packet is not None:
                print(f"    latest frame age={packet.age_s * 1000:.0f}ms "
                      f"scale_from_capture={packet.scale_from_capture}")

            if seconds is not None and (time.time() - start_time) >= seconds:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if not core.get_status().healthy:
            exit_code = 1
        core.stop()
    return exit_code


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="SentinelSwarm-3D camera-only diagnostic (no perception, no motors)"
    )
    parser.add_argument("--config", default="config/system.yaml")
    parser.add_argument("--seconds", type=float, default=None)
    args = parser.parse_args()

    cfg = load_system_config(args.config)
    configure_logging(
        level=get(cfg, "system.log_level", "INFO"),
        robot_id=get(cfg, "robots.this_robot", "leader"),
        log_dir=get(cfg, "system.log_dir"),
    )
    return run_diagnostic(cfg, seconds=args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
