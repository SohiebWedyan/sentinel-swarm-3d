"""Unit tests for runtime.process_manager.ProcessManager.

Uses real multiprocessing.Process instances (module-level target functions,
since multiprocessing needs picklable targets) rather than mocking, so
these tests exercise the actual start/stop/health/restart/disabled-process
behavior the leader-only runtime depends on.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from runtime.process_manager import (
    MODE_LEADER_ONLY,
    MODE_SWARM,
    ProcessManager,
    SwarmProcessRefused,
)


def _sleep_forever(stop_event) -> None:
    while not stop_event.is_set():
        time.sleep(0.05)


def _exit_immediately() -> None:
    return


def test_enabled_process_starts_and_is_healthy():
    manager = ProcessManager(health_check_interval_s=0.2)
    stop_event = mp.Event()
    manager.register("worker", target=_sleep_forever, args=(stop_event,), enabled=True)

    manager.start_all()
    try:
        time.sleep(0.2)
        assert manager.is_healthy("worker") is True
        assert manager.status()["worker"]["alive"] is True
    finally:
        stop_event.set()
        manager.stop_all()


def test_disabled_process_never_starts():
    manager = ProcessManager(health_check_interval_s=0.2)
    stop_event = mp.Event()
    manager.register("worker", target=_sleep_forever, args=(stop_event,), enabled=False)

    manager.start_all()
    try:
        time.sleep(0.2)
        assert manager.is_healthy("worker") is False
        assert manager.status()["worker"]["enabled"] is False
        assert manager.status()["worker"]["alive"] is False
    finally:
        stop_event.set()
        manager.stop_all()


def test_stop_all_terminates_running_processes():
    manager = ProcessManager(health_check_interval_s=0.2)
    stop_event = mp.Event()
    manager.register("worker", target=_sleep_forever, args=(stop_event,), enabled=True)

    manager.start_all()
    time.sleep(0.2)
    assert manager.is_healthy("worker") is True

    stop_event.set()
    manager.stop_all()

    assert manager.is_healthy("worker") is False


def test_crashed_process_is_restarted_up_to_max_restarts():
    manager = ProcessManager(health_check_interval_s=0.1)
    manager.register("flaky", target=_exit_immediately, enabled=True, max_restarts=2)

    manager.start_all()
    try:
        # Give the monitor loop time to notice the exit and restart a
        # couple of times, then give up once max_restarts is exceeded.
        time.sleep(1.0)
        status = manager.status()["flaky"]
        assert status["restart_count"] == 2
    finally:
        manager.stop_all()


def test_register_duplicate_name_raises():
    manager = ProcessManager()
    manager.register("worker", target=_exit_immediately, enabled=False)
    try:
        manager.register("worker", target=_exit_immediately, enabled=False)
        assert False, "expected ValueError for duplicate registration"
    except ValueError:
        pass


# -- leader_only mode enforcement -----------------------------------------------------


def test_leader_only_mode_is_the_default():
    assert ProcessManager().mode == MODE_LEADER_ONLY


def test_swarm_process_is_refused_in_leader_only_mode():
    manager = ProcessManager(mode=MODE_LEADER_ONLY)
    with pytest.raises(SwarmProcessRefused):
        manager.register("formation_process", target=_exit_immediately, requires_swarm=True)

    assert "formation_process" not in manager.status()


def test_swarm_process_is_allowed_once_the_manager_is_in_swarm_mode():
    manager = ProcessManager(mode=MODE_SWARM)
    manager.register("formation_process", target=_exit_immediately,
                     enabled=False, requires_swarm=True)

    assert manager.status()["formation_process"]["requires_swarm"] is True


def test_non_swarm_processes_register_normally_in_leader_only_mode():
    manager = ProcessManager(mode=MODE_LEADER_ONLY)
    manager.register("lidar_process", target=_exit_immediately, enabled=False)

    assert manager.status()["lidar_process"]["requires_swarm"] is False
