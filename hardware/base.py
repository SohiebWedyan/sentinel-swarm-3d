"""Base interfaces for the Hardware Layer.

Every physical sensor (camera, IMU, GPS, wheel encoders) implements
BaseSensor so the layers above can treat "no hardware attached" as a normal,
handled case (coding rule #9: handle hardware absence gracefully) instead of
an exception that takes down the process.

The LiDAR is the exception: it uses its own BaseLidar interface
(hardware/lidar/base_lidar.py) with initialize/start/stop/shutdown, because
a spinning LiDAR has a meaningful "connected but not scanning" state that
open/read/close cannot express.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class HardwareUnavailableError(RuntimeError):
    """Raised by a sensor's open()/read() when the underlying device is
    missing or fails to initialize. Callers are expected to catch this and
    fall back (simulation, degraded mode, or skip) rather than crash.
    """


class BaseSensor(ABC):
    """Common lifecycle for any physical or simulated sensor."""

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying device. Raise HardwareUnavailableError on failure."""

    @abstractmethod
    def read(self) -> Any:
        """Return the latest sample. Shape is sensor-specific."""

    @abstractmethod
    def is_healthy(self) -> bool:
        """Cheap liveness check used by the watchdog (main.py)."""

    @abstractmethod
    def close(self) -> None:
        """Release the device. Must be safe to call even if open() failed."""

    def __enter__(self) -> "BaseSensor":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
