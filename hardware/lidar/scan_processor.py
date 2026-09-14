"""Scan filtering/cleanup — the scan_processing_thread's work.

    Raw SDK scan (already a LidarScanPacket from the driver)
        -> drop non-finite / zero returns
        -> range filter against the X4 Pro's 0.12-10.0 m window
        -> angle normalization to signed degrees in [-180, 180)
        -> optional, conservative noise filtering
        -> processed LidarScanPacket (valid_points / invalid_points filled in)

Works on any LidarScanPacket regardless of which BaseLidar produced it, so
MockLidar and YDLidarX4ProDriver go through exactly the same pipeline.

A note on not over-filtering: the noise filter only removes a point that
disagrees with BOTH of its circular neighbors. A real obstacle edge (a wall
corner, the near face of a box) steps on one side only, so it survives.
That matters — an over-eager filter here would delete precisely the returns
the safety layer exists to see. It can be turned off entirely via
config/lidar.yaml's filtering.enable_noise_filter.
"""

from __future__ import annotations

import logging
import math

from common.types import LidarPoint, LidarScanPacket

logger = logging.getLogger(__name__)


class ScanProcessor:
    def __init__(
        self,
        min_distance_m: float = 0.12,
        max_distance_m: float = 10.0,
        max_neighbor_jump_m: float = 1.5,
        enable_noise_filter: bool = True,
    ) -> None:
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        self.max_neighbor_jump_m = max_neighbor_jump_m
        self.enable_noise_filter = enable_noise_filter

    def process(self, scan: LidarScanPacket) -> LidarScanPacket:
        """Return a new packet holding only valid points, with
        valid_points/invalid_points recording what happened. Never mutates
        the input (the raw packet stays available for diagnostics).
        """
        raw_count = len(scan.points)

        in_range = [p for p in scan.points if self._is_valid_distance(p.distance_m)]
        normalized = [self._normalize_angle(p) for p in in_range]
        normalized.sort(key=lambda p: p.angle_deg)

        cleaned = self._reject_isolated_spikes(normalized) if self.enable_noise_filter else normalized

        return LidarScanPacket(
            points=cleaned,
            timestamp=scan.timestamp,
            sequence_id=scan.sequence_id,
            frame_id=scan.frame_id,
            source=scan.source,
            scan_duration_s=scan.scan_duration_s,
            scan_frequency_hz=scan.scan_frequency_hz,
            valid_points=len(cleaned),
            invalid_points=raw_count - len(cleaned),
        )

    def _is_valid_distance(self, distance_m: float) -> bool:
        if distance_m is None or math.isnan(distance_m) or math.isinf(distance_m):
            return False
        return self.min_distance_m <= distance_m <= self.max_distance_m

    @staticmethod
    def _normalize_angle(point: LidarPoint) -> LidarPoint:
        """Idempotent: drivers already emit [-180, 180), but a scan that
        arrives in 0-360 form (a different driver, a replayed log) is
        corrected here rather than silently landing in the wrong sector.
        """
        angle = ((point.angle_deg + 180.0) % 360.0) - 180.0
        if angle == point.angle_deg:
            return point
        return LidarPoint(
            angle_deg=angle,
            distance_m=point.distance_m,
            intensity=point.intensity,
            timestamp=point.timestamp,
        )

    def _reject_isolated_spikes(self, points: list[LidarPoint]) -> list[LidarPoint]:
        """Drop a point only when it disagrees with BOTH circular neighbors
        by more than max_neighbor_jump_m — a lone bad return surrounded by
        consistent readings, never a genuine one-sided step.
        """
        n = len(points)
        if n < 3:
            return points

        keep = [True] * n
        for i in range(n):
            prev_p = points[i - 1]
            next_p = points[(i + 1) % n]
            cur = points[i]
            jump_prev = abs(cur.distance_m - prev_p.distance_m)
            jump_next = abs(cur.distance_m - next_p.distance_m)
            if jump_prev > self.max_neighbor_jump_m and jump_next > self.max_neighbor_jump_m:
                keep[i] = False

        filtered = [p for p, k in zip(points, keep) if k]
        if len(filtered) < n:
            logger.debug("ScanProcessor: rejected %d isolated-spike point(s)", n - len(filtered))
        return filtered


def build_scan_processor_from_config(cfg: dict) -> ScanProcessor:
    l_cfg = cfg["lidar"]
    filtering_cfg = l_cfg.get("filtering", {})
    return ScanProcessor(
        min_distance_m=l_cfg.get("min_distance", l_cfg.get("min_distance_m", 0.12)),
        max_distance_m=l_cfg.get("max_distance", l_cfg.get("max_distance_m", 10.0)),
        max_neighbor_jump_m=filtering_cfg.get("max_neighbor_jump_m", 1.5),
        enable_noise_filter=filtering_cfg.get("enable_noise_filter", True),
    )
