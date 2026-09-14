"""LiDAR process.

Isolated from perception/navigation so a LiDAR hang, USB disconnect, or
SDK crash can't stall anything else, and so ProcessManager's health-check/
restart logic can recover it independently of the rest of the system.

Exactly two internal threads:

    lidar_scan_thread      -- acquires scans from the BaseLidar driver,
                              timestamps them, updates the latest raw scan
    scan_processing_thread -- validates/range-filters/normalizes them
                              (ScanProcessor), computes obstacle sectors
                              (ObstacleSectorAnalyzer), publishes the result

Both hand-offs use a bounded, latest-only buffer (maxsize=1, oldest
dropped): scans are never allowed to accumulate, and a slow consumer costs
freshness rather than memory.

LidarProcessCore is usable two ways:
  * directly, in-process (unit tests, the diagnostic CLI at the bottom of
    this file) via start()/get_latest_scan()/get_sector_clearances()/stop()
  * as the body of a real OS process via run_lidar_process_entrypoint(),
    which is what ProcessManager spawns, publishing results back to the
    parent through runtime.shared_state.LatestValueQueue.

Run the diagnostic directly with:
    python -m runtime.lidar_process --seconds 10
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from common.config_loader import get, load_system_config
from common.logging_utils import configure_logging
from common.types import (
    LidarSafetyLevel,
    LidarScanPacket,
    LidarSectorName,
    LidarStatus,
    SectorClearance,
)
from hardware.lidar import build_lidar_from_config
from hardware.lidar.base_lidar import BaseLidar
from hardware.lidar.obstacle_sectors import ObstacleSectorAnalyzer, build_sector_analyzer_from_config
from hardware.lidar.scan_processor import ScanProcessor, build_scan_processor_from_config
from runtime.shared_state import LatestValueQueue

logger = logging.getLogger(__name__)


class LidarProcessCore:
    def __init__(
        self,
        lidar: BaseLidar,
        processor: ScanProcessor,
        analyzer: ObstacleSectorAnalyzer,
        scan_timeout_s: float = 2.0,
    ) -> None:
        self.lidar = lidar
        self.processor = processor
        self.analyzer = analyzer
        self.scan_timeout_s = scan_timeout_s

        self._raw_scan_queue: "queue.Queue[LidarScanPacket]" = queue.Queue(maxsize=1)
        self._latest_filtered_scan: Optional[LidarScanPacket] = None
        self._latest_sectors: Optional[dict[LidarSectorName, SectorClearance]] = None
        self._latest_safety_level: LidarSafetyLevel = LidarSafetyLevel.NO_DATA
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._scan_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Bring the device up and start both threads.

        initialize()/start() never raise on a dead device — the driver
        records the failure in its status and reports unhealthy — so the
        threads start regardless and the process stays alive to report the
        problem (and to recover, if auto_reconnect is on).
        """
        self.lidar.initialize()
        self.lidar.start()

        status = self.lidar.get_status()
        if not status.connected:
            logger.error("lidar_process starting with an unavailable LiDAR: %s",
                         status.error_message or "unknown error")

        self._stop_event.clear()
        self._scan_thread = threading.Thread(
            target=self._lidar_scan_loop, daemon=True, name="lidar_scan_thread"
        )
        self._process_thread = threading.Thread(
            target=self._scan_processing_loop, daemon=True, name="scan_processing_thread"
        )
        self._scan_thread.start()
        self._process_thread.start()

    def _lidar_scan_loop(self) -> None:
        """Acquisition only — no filtering or analysis on this thread, so a
        slow analysis pass can never cause a scan read to be missed.
        """
        while not self._stop_event.is_set():
            try:
                scan = self.lidar.get_scan(timeout_s=0.5)
            except Exception as exc:  # noqa: BLE001 - a driver bug must not kill the process
                logger.error("lidar_scan_thread: unexpected driver error: %s", exc)
                self._stop_event.wait(0.25)
                continue

            if scan is None:
                continue
            self._put_latest_raw(scan)

    def _put_latest_raw(self, scan: LidarScanPacket) -> None:
        """Bounded latest-only hand-off: drop the stale scan, keep the new."""
        try:
            self._raw_scan_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._raw_scan_queue.put_nowait(scan)
        except queue.Full:
            pass

    def _scan_processing_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw_scan = self._raw_scan_queue.get(timeout=0.5)
            except queue.Empty:
                self._expire_stale_results()
                continue

            try:
                filtered = self.processor.process(raw_scan)
                sectors = self.analyzer.analyze(filtered)
                safety_level = self.analyzer.get_front_safety_level(sectors)
                with self._lock:
                    self._latest_filtered_scan = filtered
                    self._latest_sectors = sectors
                    self._latest_safety_level = safety_level
            except Exception as exc:  # noqa: BLE001
                logger.error("scan_processing_thread: processing failed: %s", exc)

    def _expire_stale_results(self) -> None:
        """Stop presenting old obstacle data once it's past the scan timeout.

        Without this, a LiDAR that dies would leave its last good scan
        sitting in the published state looking current — the exact failure
        mode where a consumer keeps trusting a sensor that stopped
        reporting.
        """
        with self._lock:
            scan = self._latest_filtered_scan
            if scan is None:
                return
            if (time.time() - scan.timestamp) > self.scan_timeout_s:
                self._latest_filtered_scan = None
                self._latest_sectors = None
                self._latest_safety_level = LidarSafetyLevel.NO_DATA
                logger.warning("No LiDAR scan within %.1fs — clearing stale obstacle data.",
                               self.scan_timeout_s)

    def get_latest_scan(self) -> Optional[LidarScanPacket]:
        with self._lock:
            return self._latest_filtered_scan

    def get_sector_clearances(self) -> Optional[dict[LidarSectorName, SectorClearance]]:
        with self._lock:
            return self._latest_sectors

    def get_safety_level(self) -> LidarSafetyLevel:
        with self._lock:
            return self._latest_safety_level

    def is_healthy(self) -> bool:
        return self.lidar.is_healthy()

    def get_status(self) -> LidarStatus:
        return self.lidar.get_status()

    def stop(self) -> None:
        self._stop_event.set()
        for t in (self._scan_thread, self._process_thread):
            if t is not None:
                t.join(timeout=2.0)
        self.lidar.stop()
        self.lidar.shutdown()


def build_lidar_process_core(cfg: dict) -> LidarProcessCore:
    """Assemble driver + processor + analyzer from config. Shared by the
    process entrypoint and the diagnostic CLI so both exercise identical
    wiring.
    """
    return LidarProcessCore(
        lidar=build_lidar_from_config(cfg),
        processor=build_scan_processor_from_config(cfg),
        analyzer=build_sector_analyzer_from_config(cfg),
        scan_timeout_s=get(cfg, "lidar.scan_timeout", 2.0),
    )


def run_lidar_process_entrypoint(
    cfg: dict,
    scan_output: LatestValueQueue,
    health_output: LatestValueQueue,
    stop_event,
    publish_period_s: float = 0.05,
) -> None:
    """multiprocessing.Process target. Runs LidarProcessCore until
    stop_event (a multiprocessing.Event) is set, publishing
    (filtered_scan, sectors, safety_level) and a LidarStatus to the parent
    process via the given LatestValueQueue instances.

    Re-runs configure_logging() because this executes in a child process —
    on a fork start method (the default on Linux/Jetson) the parent's
    logging handlers are inherited anyway, but re-configuring here keeps
    this correct if the start method is ever changed to spawn.

    Also resets SIGINT/SIGTERM to their default disposition. On fork, this
    process otherwise inherits run_leader.py's custom handler (which raises
    an exception to unwind its own main loop) — ProcessManager.stop()'s
    process.terminate() would then land on that inherited handler instead
    of cleanly killing the process, producing a spurious traceback on every
    shutdown instead of a clean exit. The stop_event-driven loop below is
    already this process's graceful-shutdown path; a signal should only
    ever mean "die now."
    """
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    configure_logging(
        level=get(cfg, "system.log_level", "INFO"),
        robot_id=get(cfg, "robots.this_robot", "leader"),
        log_dir=get(cfg, "system.log_dir"),
    )

    health_check_interval_s = get(cfg, "lidar.health_check_interval", 1.0)

    try:
        core = build_lidar_process_core(cfg)
    except Exception as exc:  # noqa: BLE001 - e.g. LidarConfigurationError
        logger.error("lidar_process could not be built: %s", exc)
        # put_latest_sync, not put_latest: this is the last thing this
        # process says before exiting, and an async queue write would
        # very likely be lost — leaving the parent with a dead LiDAR and
        # no explanation.
        health_output.put_latest_sync(LidarStatus(error_message=str(exc)))
        return

    logger.info("lidar_process starting (simulation_mode=%s, driver=%s)",
                get(cfg, "lidar.simulation_mode", False),
                get(cfg, "lidar.driver", "ydlidar_x4_pro"))
    core.start()

    last_health_publish = 0.0
    last_logged_health: Optional[bool] = None
    process_start = time.time()
    # Before the first scan arrives, "unhealthy" is just "not up yet" — the
    # scan timeout hasn't elapsed, so warning about it would fire on every
    # normal boot and train operators to ignore the message that matters.
    startup_grace_s = get(cfg, "lidar.scan_timeout", 2.0)

    try:
        while not stop_event.is_set():
            scan = core.get_latest_scan()
            if scan is not None:
                scan_output.put_latest((scan, core.get_sector_clearances(), core.get_safety_level()))

            now = time.time()
            if (now - last_health_publish) >= health_check_interval_s:
                status = core.get_status()
                health_output.put_latest(status)
                last_health_publish = now

                still_starting_up = (
                    status.last_scan_timestamp is None
                    and (now - process_start) < startup_grace_s
                )
                health_changed = last_logged_health is None or status.healthy != last_logged_health

                if health_changed and not still_starting_up:
                    if status.healthy:
                        logger.info("LiDAR healthy | model=%s port=%s rate=%.1fHz",
                                    status.model, status.port, status.scan_rate_hz)
                    else:
                        reason = status.error_message or (
                            "no scan received since startup" if status.last_scan_timestamp is None
                            else "no scans within scan_timeout"
                        )
                        logger.warning("LiDAR unhealthy | model=%s port=%s | %s",
                                       status.model, status.port, reason)
                    last_logged_health = status.healthy

            time.sleep(publish_period_s)
    finally:
        core.stop()
        logger.info("lidar_process stopped")


# ---------------------------------------------------------------------------
# LiDAR-only diagnostic mode: brings up just the LiDAR (no process manager,
# no other subsystem) and prints scan/sector/health information. This is the
# first thing to run when bringing up real X4 Pro hardware.
# ---------------------------------------------------------------------------


def run_diagnostic(cfg: dict, seconds: Optional[float] = None, print_period_s: float = 1.0) -> int:
    core = build_lidar_process_core(cfg)
    core.start()

    start_time = time.time()
    exit_code = 0

    try:
        while True:
            time.sleep(print_period_s)

            status = core.get_status()
            scan = core.get_latest_scan()
            sectors = core.get_sector_clearances()

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"model={status.model} port={status.port} baud={status.baudrate} "
                f"connected={status.connected} scanning={status.scanning} "
                f"healthy={status.healthy} "
                f"rate={status.scan_rate_hz:.1f}Hz (expected {status.expected_scan_rate_hz:.1f}Hz) "
                f"simulated={status.simulated}"
            )
            if status.error_message:
                print(f"    error: {status.error_message}")

            if scan is not None:
                print(f"    scan #{scan.sequence_id}: {scan.valid_points} valid, "
                      f"{scan.invalid_points} rejected, {scan.scan_frequency_hz:.1f}Hz")
            if sectors:
                summary = "  ".join(
                    f"{name.value}={'n/a' if c.min_distance_m is None else format(c.min_distance_m, '.2f')}"
                    for name, c in sectors.items()
                )
                print(f"    clearances(m): {summary}")
                print(f"    front safety level: {core.get_safety_level().value}")

            if seconds is not None and (time.time() - start_time) >= seconds:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        final_status = core.get_status()
        if not final_status.healthy:
            exit_code = 1
        core.stop()

    return exit_code


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="SentinelSwarm-3D LiDAR-only diagnostic (no motors, no other subsystems)"
    )
    parser.add_argument("--config", default="config/system.yaml", help="Path to system.yaml")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Run for this many seconds (default: until Ctrl+C)")
    parser.add_argument("--simulation", action="store_true",
                        help="Force simulation_mode (MockLidar) regardless of config")
    args = parser.parse_args()

    cfg = load_system_config(args.config)
    if args.simulation:
        cfg.setdefault("lidar", {})["simulation_mode"] = True

    configure_logging(
        level=get(cfg, "system.log_level", "INFO"),
        robot_id=get(cfg, "robots.this_robot", "leader"),
        log_dir=get(cfg, "system.log_dir"),
    )

    return run_diagnostic(cfg, seconds=args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
