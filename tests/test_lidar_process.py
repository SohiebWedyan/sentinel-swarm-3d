"""Integration tests for runtime/lidar_process.py.

Covers process startup in simulation mode — both in-process
(LidarProcessCore, its two threads, the bounded hand-off) and as a real
multiprocessing child supervised by ProcessManager, which is exactly how
run_leader.py starts it.

No hardware and no YDLiDAR SDK required.
"""

from __future__ import annotations

import multiprocessing as mp
import time

from common.types import LidarSafetyLevel, LidarStatus
from runtime.lidar_process import (
    LidarProcessCore,
    build_lidar_process_core,
    run_lidar_process_entrypoint,
)
from runtime.process_manager import MODE_LEADER_ONLY, ProcessManager
from runtime.shared_state import LatestValueQueue


def _sim_cfg() -> dict:
    return {
        "system": {"log_level": "WARNING"},
        "robots": {"this_robot": "leader"},
        "safety": {
            "caution_distance_m": 0.8,
            "stop_distance_m": 0.35,
            "emergency_distance_m": 0.15,
        },
        "lidar": {
            "enabled": True,
            "driver": "ydlidar_x4_pro",
            "simulation_mode": True,
            "expected_scan_frequency_hz": 20.0,
            "min_distance": 0.12,
            "max_distance": 10.0,
            "frame_id": "lidar",
            "scan_timeout": 2.0,
            "health_check_interval": 0.1,
            "filtering": {"enable_noise_filter": True, "max_neighbor_jump_m": 1.5},
            "obstacle_sectors": {
                "min_points_for_valid_sector": 1,
                "obstacle_distance_m": 0.8,
            },
            "simulation": {"points_per_scan": 180},
        },
    }


def _wait_for(predicate, timeout_s: float = 5.0, interval_s: float = 0.05) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


# -- LidarProcessCore, in-process -----------------------------------------------------


def test_core_starts_both_threads_and_produces_processed_scans():
    core: LidarProcessCore = build_lidar_process_core(_sim_cfg())
    core.start()
    try:
        assert _wait_for(lambda: core.get_latest_scan() is not None)

        scan = core.get_latest_scan()
        assert scan is not None
        assert scan.valid_points > 0
        assert core.get_sector_clearances() is not None
        assert core.is_healthy() is True

        status = core.get_status()
        assert status.simulated is True
        assert status.healthy is True
    finally:
        core.stop()


def test_core_publishes_a_front_safety_level():
    core = build_lidar_process_core(_sim_cfg())
    core.start()
    try:
        assert _wait_for(lambda: core.get_latest_scan() is not None)
        assert core.get_safety_level() in set(LidarSafetyLevel)
    finally:
        core.stop()


def test_core_stops_cleanly_and_reports_unhealthy_afterwards():
    core = build_lidar_process_core(_sim_cfg())
    core.start()
    assert _wait_for(lambda: core.get_latest_scan() is not None)

    core.stop()

    assert core.is_healthy() is False


def test_core_clears_stale_obstacle_data_after_scan_timeout():
    """A LiDAR that goes quiet must not leave its last scan sitting in the
    published state looking current.
    """
    cfg = _sim_cfg()
    cfg["lidar"]["scan_timeout"] = 0.2
    core = build_lidar_process_core(cfg)
    core.start()
    try:
        assert _wait_for(lambda: core.get_latest_scan() is not None)

        # Stop the sensor underneath the process, leaving the last good scan.
        core.lidar.stop()

        assert _wait_for(lambda: core.get_latest_scan() is None, timeout_s=3.0)
        assert core.get_sector_clearances() is None
        assert core.get_safety_level() is LidarSafetyLevel.NO_DATA
    finally:
        core.stop()


# -- As a supervised multiprocessing child (how run_leader.py starts it) -------------------


def test_lidar_process_starts_under_process_manager_in_simulation_mode():
    manager = ProcessManager(health_check_interval_s=0.2, mode=MODE_LEADER_ONLY)
    scan_output: LatestValueQueue = LatestValueQueue()
    health_output: LatestValueQueue = LatestValueQueue()
    stop_event = mp.Event()

    manager.register(
        "lidar_process",
        target=run_lidar_process_entrypoint,
        args=(_sim_cfg(), scan_output, health_output, stop_event),
        enabled=True,
    )
    manager.start_all()

    try:
        assert _wait_for(lambda: manager.is_healthy("lidar_process"), timeout_s=5.0)

        published = {}

        def _collect() -> bool:
            """Keep taking the newest published values until a scan has
            arrived AND the status has settled to healthy.

            Statuses are published on a fixed interval starting immediately,
            so the first one legitimately predates the first scan and reads
            unhealthy — latching that sample would be testing the startup
            transient rather than the steady state.
            """
            scan_msg = scan_output.get_latest()
            status_msg = health_output.get_latest()
            if scan_msg is not None:
                published["scan"] = scan_msg
            if status_msg is not None:
                published["status"] = status_msg
            return "scan" in published and published.get("status") is not None \
                and published["status"].healthy

        assert _wait_for(_collect, timeout_s=8.0), \
            f"lidar_process never reached a healthy published state: {published.get('status')}"

        scan, sectors, safety_level = published["scan"]
        assert scan.valid_points > 0
        assert sectors is not None
        assert safety_level in set(LidarSafetyLevel)

        status: LidarStatus = published["status"]
        assert status.simulated is True
        assert status.healthy is True
    finally:
        stop_event.set()
        manager.stop_all()


def test_lidar_process_survives_an_invalid_configuration_without_crashing_the_runtime():
    """A bad config must surface as an unhealthy LiDAR with an error
    message, not as an exception that takes the Leader runtime down.
    """
    cfg = _sim_cfg()
    cfg["lidar"]["simulation_mode"] = False
    cfg["lidar"]["baudrate"] = 115200          # invalid for the X4 Pro

    scan_output: LatestValueQueue = LatestValueQueue()
    health_output: LatestValueQueue = LatestValueQueue()
    stop_event = mp.Event()
    stop_event.set()   # entrypoint should bail out before the loop anyway

    run_lidar_process_entrypoint(cfg, scan_output, health_output, stop_event)

    status = health_output.get_latest()
    assert status is not None
    assert status.healthy is False
    assert "128000" in (status.error_message or "")
