"""LiDAR-based obstacle detection via configurable angular sectors.

Reduces a filtered LidarScanPacket to one SectorClearance per named sector
(FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, RIGHT, REAR by default; the angle
ranges are config-driven). This is what the future safety controller
(Phase 1G) and reactive obstacle avoidance (Phase 1H) will query, via
get_front/left/right/rear_clearance(), instead of touching raw points.

It also bands the front clearance into a LidarSafetyLevel using the
thresholds in config/safety.yaml. That classification is PUBLISHED ONLY —
no motor-control code exists in this repository yet, and nothing here
commands motion. It exists so the safety layer, when it is written,
consumes a value that has already been defined and tested rather than
inventing one at the moment it starts driving motors.
"""

from __future__ import annotations

import statistics
from typing import Optional

from common.types import LidarSafetyLevel, LidarScanPacket, LidarSectorName, SectorClearance

#: Default sector boundaries in signed degrees, 0 = forward, positive = left
#: (counter-clockwise). REAR wraps across +/-180 so it's two ranges.
DEFAULT_SECTOR_RANGES_DEG: dict[LidarSectorName, list[tuple[float, float]]] = {
    LidarSectorName.FRONT: [(-30.0, 30.0)],
    LidarSectorName.FRONT_LEFT: [(30.0, 75.0)],
    LidarSectorName.FRONT_RIGHT: [(-75.0, -30.0)],
    LidarSectorName.LEFT: [(75.0, 105.0)],
    LidarSectorName.RIGHT: [(-105.0, -75.0)],
    LidarSectorName.REAR: [(105.0, 180.0), (-180.0, -105.0)],
}


class SafetyThresholds:
    """Distance bands for the front sector, in meters.

    Sourced from config/safety.yaml so there is one definition of "too
    close" in the project. (config/lidar.yaml previously carried its own
    duplicate obstacle threshold; obstacle_distance_m there now means only
    "flag obstacle_detected in the sector summary" and defaults to the
    caution distance.)
    """

    def __init__(
        self,
        caution_distance_m: float = 0.8,
        stop_distance_m: float = 0.35,
        emergency_distance_m: float = 0.15,
    ) -> None:
        if not (emergency_distance_m < stop_distance_m < caution_distance_m):
            raise ValueError(
                "safety distances must satisfy emergency < stop < caution, got "
                f"emergency={emergency_distance_m} stop={stop_distance_m} "
                f"caution={caution_distance_m}"
            )
        self.caution_distance_m = caution_distance_m
        self.stop_distance_m = stop_distance_m
        self.emergency_distance_m = emergency_distance_m

    def classify(self, clearance_m: Optional[float]) -> LidarSafetyLevel:
        """Map a clearance to a safety band. None (no valid returns in the
        sector) is NO_DATA, never NORMAL — a blind sensor must not read as
        a clear path.
        """
        if clearance_m is None:
            return LidarSafetyLevel.NO_DATA
        if clearance_m <= self.emergency_distance_m:
            return LidarSafetyLevel.EMERGENCY
        if clearance_m <= self.stop_distance_m:
            return LidarSafetyLevel.STOP
        if clearance_m <= self.caution_distance_m:
            return LidarSafetyLevel.CAUTION
        return LidarSafetyLevel.NORMAL


class ObstacleSectorAnalyzer:
    def __init__(
        self,
        sector_ranges_deg: Optional[dict[LidarSectorName, list[tuple[float, float]]]] = None,
        obstacle_distance_m: float = 0.8,
        min_points_for_valid_sector: int = 3,
        safety_thresholds: Optional[SafetyThresholds] = None,
    ) -> None:
        self.sector_ranges_deg = sector_ranges_deg or DEFAULT_SECTOR_RANGES_DEG
        self.obstacle_distance_m = obstacle_distance_m
        self.min_points_for_valid_sector = min_points_for_valid_sector
        self.safety_thresholds = safety_thresholds or SafetyThresholds()

    def analyze(self, scan: LidarScanPacket) -> dict[LidarSectorName, SectorClearance]:
        result: dict[LidarSectorName, SectorClearance] = {}
        for sector, ranges in self.sector_ranges_deg.items():
            distances = [
                p.distance_m for p in scan.points if self._angle_in_ranges(p.angle_deg, ranges)
            ]
            result[sector] = self._summarize(sector, distances)
        return result

    def _angle_in_ranges(self, angle_deg: float, ranges: list[tuple[float, float]]) -> bool:
        return any(lo <= angle_deg <= hi for lo, hi in ranges)

    def _summarize(self, sector: LidarSectorName, distances: list[float]) -> SectorClearance:
        if len(distances) < self.min_points_for_valid_sector:
            return SectorClearance(
                sector=sector,
                min_distance_m=None,
                median_distance_m=None,
                num_valid_points=len(distances),
                obstacle_detected=False,
            )

        min_distance = min(distances)
        return SectorClearance(
            sector=sector,
            min_distance_m=min_distance,
            median_distance_m=statistics.median(distances),
            num_valid_points=len(distances),
            obstacle_detected=min_distance <= self.obstacle_distance_m,
        )

    # -- clearance queries for the future safety/navigation layers ---------------------

    def get_front_clearance(self, sectors: dict[LidarSectorName, SectorClearance]) -> Optional[float]:
        return self._min_distance_or_none(sectors, LidarSectorName.FRONT)

    def get_left_clearance(self, sectors: dict[LidarSectorName, SectorClearance]) -> Optional[float]:
        return self._min_distance_or_none(sectors, LidarSectorName.LEFT)

    def get_right_clearance(self, sectors: dict[LidarSectorName, SectorClearance]) -> Optional[float]:
        return self._min_distance_or_none(sectors, LidarSectorName.RIGHT)

    def get_rear_clearance(self, sectors: dict[LidarSectorName, SectorClearance]) -> Optional[float]:
        return self._min_distance_or_none(sectors, LidarSectorName.REAR)

    def get_front_safety_level(
        self, sectors: dict[LidarSectorName, SectorClearance]
    ) -> LidarSafetyLevel:
        """Front-sector safety band. Published for monitoring//future
        consumers; nothing acts on it today.
        """
        return self.safety_thresholds.classify(self.get_front_clearance(sectors))

    def _min_distance_or_none(
        self, sectors: dict[LidarSectorName, SectorClearance], sector: LidarSectorName
    ) -> Optional[float]:
        clearance = sectors.get(sector)
        return clearance.min_distance_m if clearance else None


def build_sector_analyzer_from_config(cfg: dict) -> ObstacleSectorAnalyzer:
    l_cfg = cfg["lidar"]
    sectors_cfg = l_cfg.get("obstacle_sectors", {})
    safety_cfg = cfg.get("safety", {})

    thresholds = SafetyThresholds(
        caution_distance_m=safety_cfg.get("caution_distance_m", 0.8),
        stop_distance_m=safety_cfg.get("stop_distance_m", 0.35),
        emergency_distance_m=safety_cfg.get("emergency_distance_m", 0.15),
    )

    sector_ranges_deg: dict[LidarSectorName, list[tuple[float, float]]] = {}
    raw_sectors = sectors_cfg.get("sectors_deg")
    if raw_sectors:
        for name, ranges in raw_sectors.items():
            sector_ranges_deg[LidarSectorName(name)] = [tuple(r) for r in ranges]

    return ObstacleSectorAnalyzer(
        sector_ranges_deg=sector_ranges_deg or None,
        obstacle_distance_m=sectors_cfg.get("obstacle_distance_m", thresholds.caution_distance_m),
        min_points_for_valid_sector=sectors_cfg.get("min_points_for_valid_sector", 3),
        safety_thresholds=thresholds,
    )
