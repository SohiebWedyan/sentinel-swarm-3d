"""Perception process (Phase 2).

    FramePacket (newest only)
            |
      +-----+-----+
      |           |
    YOLO         DA3            <- BOTH on the SAME frame
      |           |
      +-----+-----+
            |
      PerceptionResult -> LatestValueQueue -> visualization / diagnostics

Both branches run against one frame, so a detection and the depth beside it
always describe the same instant. That is the whole reason this is one
process producing one result object rather than two independent streams: a
depth map from a different frame paired with a bounding box is a
mis-association that is very hard to notice downstream and very easy to
believe.

STALE FRAMES ARE DROPPED, NOT QUEUED. The input is a single-slot
LatestValueQueue, and anything older than
perception.process.stale_frame_max_age_s is discarded outright rather than
processed late — on a robot, an old answer is usually worse than no answer,
and processing a backlog only makes the next answer older still.

Depth is published with its DepthKind. Monocular DA3 is RELATIVE, not
metric; see perception/depth/da3_depth_estimator.py. Nothing here converts
it to meters or presents it as distance.

This process produces NO velocity, steering, or motor output of any kind —
it is perception only.

Run standalone:
    python -m runtime.perception_process --seconds 10
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from typing import Optional

from common.config_loader import get, load_system_config
from common.logging_utils import configure_logging
from common.types import (
    DepthFrame,
    DepthKind,
    FramePacket,
    PerceptionHealth,
    PerceptionResult,
    PerceptionTiming,
)
from perception.depth.base_depth_estimator import BaseDepthEstimator
from perception.depth.da3_depth_estimator import build_depth_estimator_from_config
from perception.detector.base_detector import BaseObjectDetector
from perception.detector.yolo_detector import build_detector_from_config
from runtime.shared_state import LatestValueQueue

logger = logging.getLogger(__name__)


class PerceptionProcessCore:
    def __init__(
        self,
        detector: BaseObjectDetector,
        depth_estimator: Optional[BaseDepthEstimator],
        stale_frame_max_age_s: float = 0.5,
        publish_frame: bool = True,
        publish_depth: bool = True,
        depth_enabled: bool = True,
        depth_every_n_frames: int = 1,
    ) -> None:
        self.detector = detector
        self.depth_estimator = depth_estimator
        self.stale_frame_max_age_s = stale_frame_max_age_s
        self.publish_frame = publish_frame
        self.publish_depth = publish_depth
        self.depth_enabled = depth_enabled and depth_estimator is not None
        # Depth runs on every Nth frame. Detection still runs on every frame,
        # so a slow depth model no longer caps detection rate.
        #
        # This does NOT weaken the pairing guarantee: a frame either carries
        # depth computed from THAT SAME frame, or carries no depth at all
        # (result.depth is None). Depth from a different frame is never
        # attached to a result.
        self.depth_every_n_frames = max(1, int(depth_every_n_frames))

        self._frames_processed = 0
        self._frames_dropped_stale = 0
        self._last_result_time: float = 0.0
        self._measured_fps: float = 0.0
        self._error_message: Optional[str] = None
        self._started = False

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Load models once, here — never per frame."""
        self.detector.initialize()
        if self.depth_enabled and self.depth_estimator is not None:
            self.depth_estimator.initialize()
        self._started = True

        logger.info("perception_process models ready | detector=%s | depth=%s",
                    self._detector_info(), self._depth_info())

        if getattr(self.detector, "is_degraded", False):
            logger.warning("YOLO detector is DEGRADED (model unavailable): no detections "
                           "will be produced.")

    def stop(self) -> None:
        self._started = False
        try:
            self.detector.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Detector shutdown failed: %s", exc)
        if self.depth_estimator is not None:
            try:
                self.depth_estimator.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Depth estimator shutdown failed: %s", exc)

    # -- per-frame processing -----------------------------------------------------

    def is_stale(self, packet: FramePacket) -> bool:
        return packet.age_s > self.stale_frame_max_age_s

    def process_frame(self, packet: FramePacket) -> Optional[PerceptionResult]:
        """Run detection + depth on one frame and build the result.

        Returns None only if the frame was dropped as stale. A failure in
        either branch degrades that branch (empty detections / no depth) but
        still produces a result, so a consumer can see WHICH half is broken
        instead of just seeing nothing.
        """
        if self.is_stale(packet):
            self._frames_dropped_stale += 1
            logger.debug("Dropping stale frame seq=%d (age %.0f ms > %.0f ms)",
                         packet.sequence_id, packet.age_s * 1000,
                         self.stale_frame_max_age_s * 1000)
            return None

        total_started = time.perf_counter()

        detections, detector_ms = self._run_detector(packet.image)

        if self._should_run_depth():
            depth_frame, depth_ms = self._run_depth(packet.image)
        else:
            depth_frame, depth_ms = None, 0.0

        total_ms = (time.perf_counter() - total_started) * 1000.0
        now = time.time()

        timing = PerceptionTiming(
            detector_ms=round(detector_ms, 2),
            depth_ms=round(depth_ms, 2),
            total_ms=round(total_ms, 2),
            # Measured from CAPTURE, not from the start of inference: this is
            # the age of the information a consumer is acting on.
            pipeline_latency_ms=round(max(0.0, now - packet.timestamp) * 1000.0, 2),
        )

        result = PerceptionResult(
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            sequence_id=packet.sequence_id,
            detections=detections,
            depth=depth_frame if self.publish_depth else None,
            image_width=packet.width,
            image_height=packet.height,
            timing=timing,
            image=packet.image if self.publish_frame else None,
        )

        self._frames_processed += 1
        if self._last_result_time > 0.0:
            delta = now - self._last_result_time
            if delta > 0:
                instantaneous = 1.0 / delta
                self._measured_fps = (
                    instantaneous if self._measured_fps == 0.0
                    else 0.8 * self._measured_fps + 0.2 * instantaneous
                )
        self._last_result_time = now
        return result

    def _should_run_depth(self) -> bool:
        """Depth cadence. Counts frames PROCESSED, not frames captured, so
        the cadence is stable regardless of how many frames were dropped.
        """
        if not self.depth_enabled:
            return False
        if self.depth_every_n_frames == 1:
            return True
        return (self._frames_processed % self.depth_every_n_frames) == 0

    def _run_detector(self, image) -> tuple[list, float]:
        started = time.perf_counter()
        try:
            detections = self.detector.infer(image)
        except Exception as exc:  # noqa: BLE001
            self._error_message = f"detector failed: {type(exc).__name__}: {exc}"
            logger.error("YOLO inference raised: %s", exc)
            detections = []
        return detections, (time.perf_counter() - started) * 1000.0

    def _run_depth(self, image) -> tuple[Optional[DepthFrame], float]:
        if not self.depth_enabled or self.depth_estimator is None:
            return None, 0.0

        started = time.perf_counter()
        try:
            depth = self.depth_estimator.infer_sync(image)
        except Exception as exc:  # noqa: BLE001
            self._error_message = f"depth failed: {type(exc).__name__}: {exc}"
            logger.error("Depth inference raised: %s", exc)
            return None, (time.perf_counter() - started) * 1000.0

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if depth is None:
            return None, elapsed_ms

        info = self.depth_estimator.get_model_info()
        return (
            DepthFrame(
                depth=depth,
                kind=self.depth_estimator.get_depth_kind(),
                model=str(info.get("model", "")),
                device=str(info.get("device", "")),
                inference_ms=round(elapsed_ms, 2),
                simulated=bool(info.get("simulated", False)),
            ),
            elapsed_ms,
        )

    # -- health -----------------------------------------------------

    def is_healthy(self) -> bool:
        """Healthy means results are actually flowing.

        A degraded detector (no model) is explicitly NOT healthy: the
        process is alive and producing empty results, and calling that
        "healthy" would hide the fact that nothing is being detected.
        """
        if not self._started or self._last_result_time == 0.0:
            return False
        if getattr(self.detector, "is_degraded", False):
            return False
        return (time.time() - self._last_result_time) <= max(2.0, self.stale_frame_max_age_s * 4)

    def _detector_info(self) -> dict:
        try:
            return self.detector.get_model_info()
        except AttributeError:
            return {}

    def _depth_info(self) -> dict:
        if self.depth_estimator is None:
            return {"backend": "disabled"}
        try:
            return self.depth_estimator.get_model_info()
        except AttributeError:
            return {}

    def get_health(self) -> PerceptionHealth:
        det_info = self._detector_info()
        depth_info = self._depth_info()
        depth_kind = DepthKind.UNKNOWN
        if self.depth_enabled and self.depth_estimator is not None:
            depth_kind = self.depth_estimator.get_depth_kind()

        return PerceptionHealth(
            healthy=self.is_healthy(),
            detector_ready=self._started and not getattr(self.detector, "is_degraded", False),
            detector_degraded=bool(getattr(self.detector, "is_degraded", False)),
            depth_ready=self.depth_enabled and self._started,
            depth_simulated=bool(depth_info.get("simulated", False)),
            depth_kind=depth_kind,
            fps=round(self._measured_fps, 2),
            last_result_timestamp=self._last_result_time or None,
            frames_processed=self._frames_processed,
            frames_dropped_stale=self._frames_dropped_stale,
            detector_model=str(det_info.get("model_path", "")),
            depth_model=str(depth_info.get("model", "")),
            device=str(det_info.get("device", "")),
            error_message=self._error_message,
        )


def build_perception_process_core(cfg: dict) -> PerceptionProcessCore:
    proc_cfg = cfg.get("perception", {}).get("process", {})
    depth_enabled = bool(get(cfg, "perception.depth.enabled", True))

    return PerceptionProcessCore(
        detector=build_detector_from_config(cfg),
        depth_estimator=build_depth_estimator_from_config(cfg) if depth_enabled else None,
        stale_frame_max_age_s=float(proc_cfg.get("stale_frame_max_age_s", 0.5)),
        publish_frame=bool(proc_cfg.get("publish_frame", True)),
        publish_depth=bool(proc_cfg.get("publish_depth", True)),
        depth_enabled=depth_enabled,
        depth_every_n_frames=int(proc_cfg.get("depth_every_n_frames", 1)),
    )


def run_perception_process_entrypoint(
    cfg: dict,
    frame_input: LatestValueQueue,
    result_output: LatestValueQueue,
    health_output: LatestValueQueue,
    stop_event,
) -> None:
    """multiprocessing.Process target.

    Pulls the newest frame, runs the pipeline, publishes the result. Blocks
    on the input queue rather than spinning, so an idle camera costs no CPU.
    """
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    configure_logging(
        level=get(cfg, "system.log_level", "INFO"),
        robot_id=get(cfg, "robots.this_robot", "leader"),
        log_dir=get(cfg, "system.log_dir"),
    )

    health_check_interval_s = float(get(cfg, "perception.process.health_check_interval", 1.0))

    try:
        core = build_perception_process_core(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.error("perception_process could not be built: %s", exc)
        health_output.put_latest_sync(PerceptionHealth(error_message=str(exc)))
        return

    logger.info("perception_process starting")
    try:
        core.start()
    except Exception as exc:  # noqa: BLE001
        logger.error("perception_process failed to initialize models: %s", exc)
        health_output.put_latest_sync(PerceptionHealth(error_message=str(exc)))
        return

    last_health_publish = 0.0
    last_logged_health: Optional[bool] = None
    process_start = time.time()
    startup_grace_s = 5.0   # model warm-up + first frame

    try:
        while not stop_event.is_set():
            packet = frame_input.get_latest_blocking(timeout=0.25)
            if packet is not None:
                try:
                    result = core.process_frame(packet)
                except Exception as exc:  # noqa: BLE001 - never let one bad frame kill the process
                    logger.error("perception_process: frame %s failed: %s",
                                 getattr(packet, "sequence_id", "?"), exc)
                    result = None
                if result is not None:
                    result_output.put_latest(result)

            now = time.time()
            if (now - last_health_publish) >= health_check_interval_s:
                health = core.get_health()
                health_output.put_latest(health)
                last_health_publish = now

                still_starting = (health.last_result_timestamp is None
                                  and (now - process_start) < startup_grace_s)
                if (last_logged_health is None or health.healthy != last_logged_health) \
                        and not still_starting:
                    if health.healthy:
                        logger.info(
                            "Perception healthy | %.1f FPS | detections/frame path active | "
                            "depth=%s (%s)",
                            health.fps,
                            "simulated" if health.depth_simulated else health.depth_model or "off",
                            health.depth_kind.value,
                        )
                    else:
                        logger.warning(
                            "Perception unhealthy | %s",
                            health.error_message or (
                                "detector degraded (no model)" if health.detector_degraded
                                else "no results within the expected interval"
                            ),
                        )
                    last_logged_health = health.healthy
    finally:
        core.stop()
        logger.info("perception_process stopped")


# ---------------------------------------------------------------------------
# Perception-only diagnostic: camera + YOLO + DA3, nothing else.
# ---------------------------------------------------------------------------


def run_diagnostic(cfg: dict, seconds: Optional[float] = None, print_period_s: float = 1.0) -> int:
    """Runs camera_process and perception_process together and prints what
    comes out. This is the Phase 2 bring-up command.
    """
    from runtime.camera_process import run_camera_process_entrypoint
    from runtime.process_manager import MODE_LEADER_ONLY, ProcessManager

    manager = ProcessManager(health_check_interval_s=1.0, mode=MODE_LEADER_ONLY)
    frame_queue: LatestValueQueue = LatestValueQueue()
    camera_status_queue: LatestValueQueue = LatestValueQueue()
    result_queue: LatestValueQueue = LatestValueQueue()
    health_queue: LatestValueQueue = LatestValueQueue()
    stop_event = mp.Event()

    manager.register("camera_process", target=run_camera_process_entrypoint,
                     args=(cfg, frame_queue, camera_status_queue, stop_event))
    manager.register("perception_process", target=run_perception_process_entrypoint,
                     args=(cfg, frame_queue, result_queue, health_queue, stop_event))
    manager.start_all()

    start_time = time.time()
    exit_code = 0
    latest_health = None
    latest_camera = None

    try:
        while True:
            time.sleep(print_period_s)
            result = result_queue.get_latest()
            health = health_queue.get_latest()
            camera_status = camera_status_queue.get_latest()
            if health is not None:
                latest_health = health
            if camera_status is not None:
                latest_camera = camera_status

            stamp = time.strftime("%H:%M:%S")
            if latest_camera is not None:
                print(f"[{stamp}] camera: {latest_camera.device} "
                      f"{latest_camera.width}x{latest_camera.height} "
                      f"healthy={latest_camera.healthy} fps={latest_camera.measured_fps:.1f} "
                      f"dropped={latest_camera.frames_dropped}")
            if latest_health is not None:
                print(f"    perception: healthy={latest_health.healthy} "
                      f"fps={latest_health.fps:.1f} "
                      f"processed={latest_health.frames_processed} "
                      f"stale_dropped={latest_health.frames_dropped_stale} "
                      f"detector_degraded={latest_health.detector_degraded} "
                      f"depth={latest_health.depth_kind.value}"
                      f"{' [SIMULATED]' if latest_health.depth_simulated else ''}")
            if result is not None:
                print(f"    frame #{result.sequence_id}: {result.detection_count} detections | "
                      f"yolo={result.timing.detector_ms:.1f}ms "
                      f"depth={result.timing.depth_ms:.1f}ms "
                      f"total={result.timing.total_ms:.1f}ms "
                      f"latency={result.timing.pipeline_latency_ms:.1f}ms")
                for det in result.detections[:5]:
                    print(f"        {det.class_name} {det.confidence:.2f} "
                          f"@({det.center_x:.0f},{det.center_y:.0f}) "
                          f"{det.width:.0f}x{det.height:.0f}")

            if seconds is not None and (time.time() - start_time) >= seconds:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if latest_health is None or not latest_health.healthy:
            exit_code = 1
        stop_event.set()
        manager.stop_all()

    return exit_code


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="SentinelSwarm-3D perception diagnostic (camera + YOLO + DA3, no motors)"
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
