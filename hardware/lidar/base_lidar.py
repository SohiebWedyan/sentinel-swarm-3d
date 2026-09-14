"""LiDAR abstraction.

Everything downstream (scan_processor, obstacle_sectors, and later
localization/mapping) depends only on this interface and on
common.types.LidarScanPacket — never on a specific LiDAR model or its SDK.

Current implementations:
    YDLidarX4ProDriver  -- the real YDLiDAR X4 Pro, via the official SDK
    MockLidar           -- synthetic scans, no hardware required

Selection between them happens in hardware/lidar/__init__.py's
build_lidar_from_config() factory, driven by config/lidar.yaml's
simulation_mode. Nothing outside hardware/lidar/ names a concrete driver
class, so adding a different sensor later is a new file plus one factory
branch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from common.types import LidarScanPacket, LidarStatus


class BaseLidar(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Connect to the device and get it ready to scan.

        Must NOT raise on a missing/failed device, and must NOT silently
        substitute fake data for real data: an implementation that cannot
        reach its hardware records the failure in get_status().error_message
        and reports is_healthy() == False, so the runtime keeps going with a
        LiDAR it knows is dead rather than a LiDAR it wrongly believes is
        alive. Simulated scans come only from explicitly selecting
        MockLidar, never as an implicit fallback.
        """

    @abstractmethod
    def start(self) -> None:
        """Begin scanning (spin up the motor / start data flow). Safe to
        call only after initialize().
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop scanning without releasing the connection — start() can be
        called again afterward.
        """

    @abstractmethod
    def get_scan(self, timeout_s: float = 1.0) -> Optional[LidarScanPacket]:
        """Block up to timeout_s for the next complete scan. Returns None on
        timeout or on a recoverable device error, rather than raising —
        callers (runtime/lidar_process.py) just loop and try again, and the
        implementation is responsible for its own reconnect policy.
        """

    @abstractmethod
    def get_latest_scan(self) -> Optional[LidarScanPacket]:
        """Return the most recently completed scan without blocking, or None
        if none is available yet.
        """

    @abstractmethod
    def is_healthy(self) -> bool:
        """True only when scans are actually arriving at roughly the
        expected rate. False whenever the device is disconnected, erroring,
        or has gone quiet past the scan timeout.
        """

    @abstractmethod
    def get_status(self) -> LidarStatus:
        """Full status snapshot for diagnostics/monitoring: connection,
        scanning state, health, port, baudrate, model, last scan time,
        measured scan rate, and the last error message if any.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Stop scanning (if running) and release the connection. Must be
        safe to call even if initialize() was never called or failed.
        """

    def __enter__(self) -> "BaseLidar":
        self.initialize()
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
