"""Generic multiprocessing process supervisor (Phase 1A).

This is the foundation the full Jetson-side process list from the spec
(camera_process, perception_process, lidar_process, fusion_process,
localization_process, mapping_process, navigation_process,
robot_bridge_process) will register with, one phase at a time, exactly the
way perception/*'s base_*.py interfaces existed before their concrete
implementations did. Only lidar_process (Phase 1B) is registered by
run_leader.py today.

Swarm-shaped processes are never registered here in leader_only mode —
enforced simply by not calling register()/start_all() for anything
swarm-related from run_leader.py, without deleting swarm/'s code. There is
currently no swarm process to register anyway (swarm/communication_client.py
is a library used synchronously, not a standalone process), but the
pattern below is what a future swarm_process would follow if one is added.

Deliberately unopinionated about IPC: each process's target function is
responsible for wiring its own runtime.shared_state.LatestValueQueue
instance(s) — this class only owns process lifecycle (start/stop) and
crash recovery (bounded restarts, never a silent infinite crash loop).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


#: Runtime modes. leader_only is the current stage: exactly one Leader UGV,
#: no followers, no formation control, no swarm communication.
MODE_LEADER_ONLY = "leader_only"
MODE_SWARM = "swarm"


class SwarmProcessRefused(RuntimeError):
    """A swarm/follower/formation process was registered while the manager
    is in leader_only mode.

    This is a hard error rather than a silent skip: if some future phase's
    wiring tries to bring up swarm behavior during the leader-only stage,
    that's a bug in the caller and should be visible immediately, not
    discovered later by noticing MQTT traffic that shouldn't exist.
    """


@dataclass
class _ManagedProcessSpec:
    name: str
    target: Callable[..., None]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    enabled: bool = True
    max_restarts: int = 3
    restart_count: int = 0
    requires_swarm: bool = False
    process: Optional[mp.process.BaseProcess] = None


class ProcessManager:
    def __init__(
        self,
        health_check_interval_s: float = 1.0,
        mode: str = MODE_LEADER_ONLY,
    ) -> None:
        self.health_check_interval_s = health_check_interval_s
        self.mode = mode
        self._specs: dict[str, _ManagedProcessSpec] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def register(
        self,
        name: str,
        target: Callable[..., None],
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        enabled: bool = True,
        max_restarts: int = 3,
        requires_swarm: bool = False,
    ) -> None:
        """Register a process by name.

        enabled=False means start_all() never launches it, but it still
        shows up in status() so it's visible that the capability exists and
        is intentionally off.

        requires_swarm=True marks a process that needs swarm/follower/
        formation behavior. In leader_only mode registering one raises
        SwarmProcessRefused — the mode is enforced here rather than relying
        on every caller to remember not to ask.
        """
        if name in self._specs:
            raise ValueError(f"Process '{name}' is already registered")

        if requires_swarm and self.mode == MODE_LEADER_ONLY:
            raise SwarmProcessRefused(
                f"Refusing to register swarm process '{name}': ProcessManager is in "
                f"{MODE_LEADER_ONLY} mode. Set leader.swarm_enabled: true in "
                f"config/leader.yaml (and construct ProcessManager with mode='{MODE_SWARM}') "
                f"once follower robots are actually part of the system."
            )

        self._specs[name] = _ManagedProcessSpec(
            name=name, target=target, args=args, kwargs=kwargs or {},
            enabled=enabled, max_restarts=max_restarts, requires_swarm=requires_swarm,
        )
        logger.info("Registered process '%s' (enabled=%s, mode=%s)", name, enabled, self.mode)

    def start_all(self) -> None:
        for spec in self._specs.values():
            if spec.enabled:
                self._start(spec)
            else:
                logger.info("Not starting '%s': disabled by configuration "
                           "(leader-only / swarm-disabled mode).", spec.name)

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="ProcessManagerMonitor"
        )
        self._monitor_thread.start()

    def _start(self, spec: _ManagedProcessSpec) -> None:
        spec.process = mp.Process(
            target=spec.target, args=spec.args, kwargs=spec.kwargs, name=spec.name, daemon=True
        )
        spec.process.start()
        logger.info("Started process '%s' (pid=%s)", spec.name, spec.process.pid)

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            for spec in self._specs.values():
                if not spec.enabled or spec.process is None:
                    continue
                if spec.process.is_alive():
                    continue

                exit_code = spec.process.exitcode
                if spec.restart_count >= spec.max_restarts:
                    logger.error(
                        "Process '%s' exited (code=%s) and has exceeded max_restarts=%d; "
                        "leaving it stopped. This does not affect any other process.",
                        spec.name, exit_code, spec.max_restarts,
                    )
                    continue

                spec.restart_count += 1
                logger.warning(
                    "Process '%s' exited unexpectedly (code=%s); restarting (%d/%d).",
                    spec.name, exit_code, spec.restart_count, spec.max_restarts,
                )
                self._start(spec)

            self._stop_event.wait(self.health_check_interval_s)

    def is_healthy(self, name: str) -> bool:
        spec = self._specs.get(name)
        return bool(spec and spec.enabled and spec.process is not None and spec.process.is_alive())

    def stop(self, name: str, timeout_s: float = 3.0) -> None:
        spec = self._specs.get(name)
        if spec is None or spec.process is None:
            return
        if spec.process.is_alive():
            spec.process.terminate()
            spec.process.join(timeout=timeout_s)
        logger.info("Stopped process '%s'", name)

    def stop_all(self, timeout_s: float = 3.0) -> None:
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)
        for name in self._specs:
            self.stop(name, timeout_s=timeout_s)

    def status(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "enabled": spec.enabled,
                "alive": bool(spec.process and spec.process.is_alive()),
                "restart_count": spec.restart_count,
                "requires_swarm": spec.requires_swarm,
            }
            for name, spec in self._specs.items()
        }
