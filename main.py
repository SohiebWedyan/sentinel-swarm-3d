#!/usr/bin/env python3
"""SentinelSwarm-3D — Phase 1 entry point.

Wires: Camera -> YOLO Detector -> DA3 Depth (async) -> 3D Object Fusion ->
Object Tracker -> Visualization Dashboard, exactly matching the spec's
perception pipeline diagram. Everything else (localization, mapping,
navigation, swarm, formation, shaper) stays off per config/system.yaml's
`layers` block — this file only drives what's actually implemented so far,
and is meant to be extended phase by phase rather than rewritten.

Usage:
    python main.py                          # uses config/system.yaml as-is
    python main.py --config config/system.yaml
    python main.py --mode live               # override system.mode
    python main.py --headless                # never touch cv2.imshow
    python main.py --max-frames 200           # run N frames then exit (CI/demo)
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from common.config_loader import get, load_system_config
from common.logging_utils import configure_logging
from hardware.base import HardwareUnavailableError
from hardware.camera import build_camera_from_config
from perception.depth.da3_depth_estimator import build_depth_estimator_from_config
from perception.detector.yolo_detector import build_detector_from_config
from perception.fusion import build_fusion_from_config
from perception.tracking.bytetrack_tracker import build_tracker_from_config
from visualization.dashboard import build_dashboard_from_config

logger = logging.getLogger("sentinelswarm.main")


class _ShutdownRequested(Exception):
    """Raised by the SIGINT/SIGTERM handler to unwind the main loop cleanly."""


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        raise _ShutdownRequested(f"signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, AttributeError):
        pass  # SIGTERM not available on this platform (e.g. some Windows setups)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SentinelSwarm-3D Phase 1 runner")
    parser.add_argument("--config", default="config/system.yaml", help="Path to system.yaml")
    parser.add_argument("--mode", choices=["simulation", "live"], default=None,
                        help="Override system.mode from the config file")
    parser.add_argument("--headless", action="store_true", help="Disable any GUI display")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Exit after this many frames (0/None = run forever)")
    return parser.parse_args()


def run(cfg: dict, headless: bool, max_frames: int | None) -> int:
    robot_id = get(cfg, "robots.this_robot", "leader")
    source_robot = get(cfg, f"robots.fleet.{robot_id}.robot_id", robot_id)

    camera = build_camera_from_config(cfg)
    detector = build_detector_from_config(cfg)
    depth_estimator = build_depth_estimator_from_config(cfg)
    fusion = build_fusion_from_config(cfg, source_robot=source_robot)
    tracker = build_tracker_from_config(cfg)
    dashboard = build_dashboard_from_config(cfg) if not headless else None

    watchdog_timeout_s = get(cfg, "system.watchdog.perception_timeout_s", 2.0)
    last_detection_time = time.time()

    logger.info("Starting SentinelSwarm-3D Phase 1 as robot_id=%s, mode=%s",
                source_robot, get(cfg, "system.mode", "simulation"))

    frame_count = 0
    fps = 0.0
    loop_start = time.time()

    try:
        camera.open()
    except HardwareUnavailableError as exc:
        logger.error("Camera unavailable (%s). Cannot continue Phase 1 without a frame "
                     "source; check config/camera.yaml source_type.", exc)
        return 1

    detector.initialize()
    depth_estimator.initialize()

    try:
        while True:
            frame_start = time.time()

            frame = camera.read()
            if frame is None:
                logger.warning("No frame available this cycle; retrying.")
                time.sleep(0.05)
                continue

            detections = detector.infer(frame.image)
            if detections:
                last_detection_time = time.time()
            elif time.time() - last_detection_time > watchdog_timeout_s:
                logger.debug("No detections for >%.1fs — this is expected in an empty "
                            "scene, but would trigger the navigation-layer degrade "
                            "policy once Phase 4 wires this watchdog up to motion.",
                            watchdog_timeout_s)

            depth_estimator.infer(frame.image)
            depth_map = depth_estimator.get_depth()
            depth_confidence = depth_estimator.get_confidence()

            objects = fusion.fuse(
                detections=detections,
                depth_map=depth_map,
                depth_confidence_map=depth_confidence,
                image_shape=frame.image.shape,
            )
            objects = tracker.update(objects)

            frame_count += 1
            elapsed_total = time.time() - loop_start
            if elapsed_total > 0:
                fps = frame_count / elapsed_total

            if dashboard is not None:
                dashboard.render(
                    frame=frame.image,
                    objects=objects,
                    depth_map=depth_map,
                    fps=fps,
                    depth_is_mock=getattr(depth_estimator, "is_mock", True),
                )

            if frame_count % 60 == 0:
                logger.info("frame=%d fps=%.1f objects_this_frame=%d depth_mock=%s",
                            frame_count, fps, len(objects), getattr(depth_estimator, "is_mock", True))

            if max_frames and frame_count >= max_frames:
                logger.info("Reached --max-frames=%d, shutting down.", max_frames)
                break

            frame_elapsed = time.time() - frame_start
            target_period = 1.0 / get(cfg, "camera.fps", 30)
            if frame_elapsed < target_period:
                time.sleep(target_period - frame_elapsed)

    except _ShutdownRequested:
        logger.info("Shutdown requested; stopping cleanly.")
    finally:
        detector.shutdown()
        depth_estimator.shutdown()
        camera.close()
        if dashboard is not None:
            dashboard.close()

    return 0


def main() -> int:
    args = parse_args()
    _install_signal_handlers()

    try:
        cfg = load_system_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Failed to load configuration: {exc}", file=sys.stderr)
        return 1

    if args.mode:
        cfg.setdefault("system", {})["mode"] = args.mode

    robot_id = get(cfg, "robots.this_robot", "leader")
    configure_logging(
        level=get(cfg, "system.log_level", "INFO"),
        robot_id=robot_id,
        log_dir=get(cfg, "system.log_dir"),
    )

    return run(cfg, headless=args.headless, max_frames=args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
