#!/usr/bin/env python3
"""SentinelSwarm-3D — Leader-only runtime entry point (Phase 1A / 1B).

This is a NEW, separate entry point from main.py. main.py's single-process
Camera -> YOLO -> DA3 -> Fusion -> Tracker -> Dashboard loop (Phase 1) is
completely unaffected by anything in this file and still runs exactly as
before.

run_leader.py boots the process supervisor (runtime/process_manager.py) in
leader_only mode: only what's implemented AND enabled in config/leader.yaml
+ config/lidar.yaml gets started. Today that is exactly one process —
lidar_process (Phase 1B). camera_process, perception_process,
fusion_process, localization_process, mapping_process, navigation_process
and robot_bridge_process from the full spec do not exist yet; they will be
registered here, one phase at a time, alongside main.py's existing
perception pipeline rather than replacing it.

No motor-control code exists anywhere in this repository yet (that is
Phase 1E/1F). There is nothing in this file, or anywhere it imports, that
can move a robot — it only proves out the process supervisor and the
LiDAR stack end-to-end, including in pure simulation with zero hardware
attached.

The LiDAR is a real YDLiDAR X4 Pro (config/lidar.yaml). If the hardware or
its SDK is unavailable the LiDAR reports itself unhealthy and publishes no
scans — it is never silently replaced with simulated data. Simulation is an
explicit choice: --simulation, or simulation_mode: true in config.

Usage:
    python run_leader.py                    # real YDLiDAR X4 Pro (per config/lidar.yaml)
    python run_leader.py --simulation        # MockLidar, no hardware touched
    python run_leader.py --headless          # no LiDAR view window
    python run_leader.py --max-seconds 20     # run for N seconds then exit (CI/demo)
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import signal
import sys
import time

from common.config_loader import get, load_system_config
from common.logging_utils import configure_logging
from common.types import LidarSafetyLevel
from runtime.camera_process import run_camera_process_entrypoint
from runtime.lidar_process import run_lidar_process_entrypoint
from runtime.perception_process import run_perception_process_entrypoint
from runtime.process_manager import MODE_LEADER_ONLY, MODE_SWARM, ProcessManager
from runtime.shared_state import LatestValueQueue
from visualization.lidar_view import LidarView
from visualization.perception_view import build_perception_view_from_config

logger = logging.getLogger("sentinelswarm.run_leader")


class _ShutdownRequested(Exception):
    """Raised by the SIGINT/SIGTERM handler to unwind the main loop cleanly."""


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        raise _ShutdownRequested(f"signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, AttributeError):
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SentinelSwarm-3D leader-only runtime (Phase 1A/1B)")
    parser.add_argument("--config", default="config/system.yaml", help="Path to system.yaml")
    parser.add_argument("--headless", action="store_true", help="Disable the LiDAR view window")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Exit after this many seconds (0/None = run forever)")
    parser.add_argument("--simulation", action="store_true",
                        help="Force MockLidar regardless of config (no hardware touched)")
    return parser.parse_args()


def _log_leader_mode(cfg: dict) -> None:
    logger.info(
        "robot_mode=%s robot_role=%s motors_enabled=%s",
        get(cfg, "leader.robot_mode", "leader_only"),
        get(cfg, "leader.robot_role", "leader"),
        get(cfg, "leader.motors_enabled", False),
    )
    logger.info(
        "Swarm disabled: swarm_enabled=%s followers_enabled=%s formation_enabled=%s "
        "global_multi_robot_map_enabled=%s shaper_enabled=%s "
        "(swarm/, shaper/, and navigation.yaml's formation config are unchanged and still "
        "importable — they are simply never started while these are false)",
        get(cfg, "leader.swarm_enabled", False),
        get(cfg, "leader.followers_enabled", False),
        get(cfg, "leader.formation_enabled", False),
        get(cfg, "leader.global_multi_robot_map_enabled", False),
        get(cfg, "leader.shaper_enabled", False),
    )


def run(cfg: dict, headless: bool, max_seconds: float | None) -> int:
    _log_leader_mode(cfg)

    mode = MODE_SWARM if get(cfg, "leader.swarm_enabled", False) else MODE_LEADER_ONLY
    manager = ProcessManager(mode=mode)
    stop_event = mp.Event()

    # One queue per producer/consumer pair. LatestValueQueue reads are
    # destructive, so two consumers must never share a queue — they would
    # steal published values from each other.
    scan_output: LatestValueQueue = LatestValueQueue()
    lidar_health_output: LatestValueQueue = LatestValueQueue()
    frame_output: LatestValueQueue = LatestValueQueue()
    camera_status_output: LatestValueQueue = LatestValueQueue()
    perception_output: LatestValueQueue = LatestValueQueue()
    perception_health_output: LatestValueQueue = LatestValueQueue()

    lidar_enabled = get(cfg, "lidar.enabled", True) and get(cfg, "system.layers.lidar", True)
    manager.register(
        "lidar_process",
        target=run_lidar_process_entrypoint,
        args=(cfg, scan_output, lidar_health_output, stop_event),
        enabled=lidar_enabled,
    )

    camera_enabled = get(cfg, "camera.enabled", True) and get(cfg, "system.layers.camera", True)
    manager.register(
        "camera_process",
        target=run_camera_process_entrypoint,
        args=(cfg, frame_output, camera_status_output, stop_event),
        enabled=camera_enabled,
    )

    perception_enabled = camera_enabled and get(cfg, "system.layers.perception", True)
    manager.register(
        "perception_process",
        target=run_perception_process_entrypoint,
        args=(cfg, frame_output, perception_output, perception_health_output, stop_event),
        enabled=perception_enabled,
    )
    # Future phases register here, following the same pattern:
    #   manager.register("fusion_process", target=..., enabled=...)
    #   manager.register("localization_process", target=..., enabled=...)
    #   manager.register("mapping_process", target=..., enabled=...)
    #   manager.register("navigation_process", target=..., enabled=...)
    #   manager.register("robot_bridge_process", target=..., enabled=...)
    # A swarm/follower/formation process would pass requires_swarm=True and
    # is hard-refused by ProcessManager while running in leader_only mode.
    #
    # The perception pipeline does NOT depend on the LiDAR: camera_process
    # and perception_process publish independently of scan_output, so a
    # degraded LiDAR cannot stop or distort perception.

    manager.start_all()

    lidar_view = LidarView(
        max_range_m=get(cfg, "lidar.max_distance", 10.0)
    ) if not headless else None
    perception_view = build_perception_view_from_config(cfg) if not headless else None
    start_time = time.time()

    latest_scan = None
    latest_sectors = None
    latest_lidar_status = None
    safety_level = LidarSafetyLevel.NO_DATA

    latest_result = None
    latest_camera_status = None
    latest_perception_health = None

    try:
        while True:
            published = scan_output.get_latest()
            if published is not None:
                latest_scan, latest_sectors, safety_level = published

            lidar_status = lidar_health_output.get_latest()
            if lidar_status is not None:
                latest_lidar_status = lidar_status

            result = perception_output.get_latest()
            if result is not None:
                latest_result = result

            camera_status = camera_status_output.get_latest()
            if camera_status is not None:
                latest_camera_status = camera_status

            perception_health = perception_health_output.get_latest()
            if perception_health is not None:
                latest_perception_health = perception_health

            if lidar_view is not None:
                lidar_view.render(latest_scan, latest_sectors, latest_lidar_status, safety_level)
            if perception_view is not None:
                perception_view.render(latest_result, latest_perception_health,
                                       latest_camera_status)

            if headless:
                if latest_scan is not None:
                    logger.debug(
                        "lidar scan #%d valid=%d healthy=%s front_safety=%s",
                        latest_scan.sequence_id, latest_scan.valid_points,
                        latest_lidar_status.healthy if latest_lidar_status else None,
                        safety_level.value,
                    )
                if latest_result is not None:
                    logger.debug(
                        "perception frame #%d detections=%d yolo=%.0fms depth=%.0fms "
                        "latency=%.0fms",
                        latest_result.sequence_id, latest_result.detection_count,
                        latest_result.timing.detector_ms, latest_result.timing.depth_ms,
                        latest_result.timing.pipeline_latency_ms,
                    )

            if max_seconds and (time.time() - start_time) >= max_seconds:
                logger.info("Reached --max-seconds=%.1f, shutting down.", max_seconds)
                break

            time.sleep(0.05)

    except _ShutdownRequested:
        logger.info("Shutdown requested; stopping cleanly.")
    finally:
        stop_event.set()
        manager.stop_all()
        if lidar_view is not None:
            lidar_view.close()
        if perception_view is not None:
            perception_view.close()

    return 0


def main() -> int:
    args = parse_args()
    _install_signal_handlers()

    try:
        cfg = load_system_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Failed to load configuration: {exc}", file=sys.stderr)
        return 1

    if args.simulation:
        cfg.setdefault("lidar", {})["simulation_mode"] = True

    robot_id = get(cfg, "robots.this_robot", "leader")
    configure_logging(
        level=get(cfg, "system.log_level", "INFO"),
        robot_id=robot_id,
        log_dir=get(cfg, "system.log_dir"),
    )

    return run(cfg, headless=args.headless, max_seconds=args.max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
