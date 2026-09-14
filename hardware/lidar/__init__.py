"""LiDAR hardware abstraction and driver factory.

    BaseLidar  ->  YDLidarX4ProDriver  ->  YDLidar SDK  ->  X4 Pro hardware
                \\-> MockLidar (simulation only, explicitly selected)

build_lidar_from_config() below is the ONLY place that decides which
concrete driver the runtime gets. Nothing outside this package imports a
driver class directly, so the rest of the system (runtime/lidar_process.py,
scan processing, sector analysis, visualization) is identical whether it's
running against real hardware or simulation.

Method naming note: BaseLidar uses initialize/start/stop/shutdown rather
than hardware/base.py's BaseSensor open/read/close, matching the style of
perception/detector/base_detector.py and
perception/depth/base_depth_estimator.py — a spinning LiDAR has a real
"connected but not spinning" state that a camera doesn't.
"""

from __future__ import annotations

import logging

from common.config_loader import get
from hardware.lidar.base_lidar import BaseLidar
from hardware.lidar.mock_lidar import MockLidar, build_mock_lidar_from_config
from hardware.lidar.ydlidar_x4_pro_driver import (
    X4_PRO,
    LidarConfigurationError,
    YDLidarX4ProDriver,
    build_ydlidar_x4_pro_from_config,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BaseLidar",
    "MockLidar",
    "YDLidarX4ProDriver",
    "LidarConfigurationError",
    "X4_PRO",
    "build_lidar_from_config",
]

#: driver name in config/lidar.yaml -> builder for the real hardware.
_REAL_DRIVER_BUILDERS = {
    "ydlidar_x4_pro": build_ydlidar_x4_pro_from_config,
}


def build_lidar_from_config(cfg: dict) -> BaseLidar:
    """Factory: config/lidar.yaml -> a concrete BaseLidar.

    simulation_mode is the ONLY thing that selects MockLidar. A real driver
    that can't reach its hardware stays a real driver and reports itself
    unhealthy — it is never swapped for MockLidar behind the operator's
    back, because synthetic obstacle data mistaken for real readings is a
    safety problem, not a convenience.
    """
    if get(cfg, "lidar.simulation_mode", False):
        logger.info("LiDAR factory: simulation_mode=true -> MockLidar (no hardware will be used)")
        return build_mock_lidar_from_config(cfg)

    driver_name = str(get(cfg, "lidar.driver", "ydlidar_x4_pro")).lower()
    builder = _REAL_DRIVER_BUILDERS.get(driver_name)
    if builder is None:
        raise LidarConfigurationError(
            f"Unknown lidar.driver '{driver_name}'. Known drivers: "
            f"{', '.join(sorted(_REAL_DRIVER_BUILDERS))}. Set lidar.simulation_mode: true "
            f"to run without hardware."
        )

    logger.info("LiDAR factory: driver=%s -> real hardware", driver_name)
    return builder(cfg)
