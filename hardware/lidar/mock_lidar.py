"""MockLidar — synthetic scans, no hardware required.

This is the simulation source migrated out of the old xplidar_driver.py
(which has been removed; its synthetic generator lives here now and its
real-hardware seam is superseded by ydlidar_x4_pro_driver.py). Behavior of
the generator is unchanged: a room of roughly fixed radius with one slowly
orbiting obstacle, so obstacle_sectors has something non-trivial to detect
during development, CI, and demos.

MockLidar is only ever used when config/lidar.yaml sets
simulation_mode: true. It is never substituted automatically for a failed
real device — see BaseLidar.initialize()'s docstring for why.

By default it mimics the YDLiDAR X4 Pro's own numbers (0.12-10.0 m,
~10 Hz, 360 points/scan) so simulation exercises roughly the same data
shape the real sensor produces.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from common.types import LidarPoint, LidarScanPacket, LidarStatus
from hardware.lidar.base_lidar import BaseLidar

logger = logging.getLogger(__name__)


class MockLidar(BaseLidar):
    def __init__(
        self,
        scan_frequency_hz: float = 10.0,
        min_distance_m: float = 0.12,
        max_distance_m: float = 10.0,
        frame_id: str = "lidar",
        points_per_scan: int = 360,
        room_radius_m: float = 4.0,
        obstacle_distance_m: float = 0.6,
        obstacle_half_width_deg: float = 12.0,
        obstacle_sweep_deg_per_s: float = 20.0,
        model: str = "MOCK",
    ) -> None:
        self.scan_frequency_hz = max(scan_frequency_hz, 0.1)
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        self.frame_id = frame_id
        self.points_per_scan = points_per_scan
        self.room_radius_m = min(room_radius_m, max_distance_m * 0.9)
        self.obstacle_distance_m = max(obstacle_distance_m, min_distance_m + 0.05)
        self.obstacle_half_width_deg = obstacle_half_width_deg
        self.obstacle_sweep_deg_per_s = obstacle_sweep_deg_per_s
        self.model = model

        self._t0 = time.time()
        self._connected = False
        self._scanning = False
        self._sequence_id = 0
        self._latest_scan: Optional[LidarScanPacket] = None
        self._last_scan_time: float = 0.0

    # -- lifecycle -----------------------------------------------------

    def initialize(self) -> None:
        self._connected = True
        self._t0 = time.time()
        logger.info("MockLidar initialized (simulation only, %.1f Hz, %d points/scan, "
                    "range %.2f-%.2fm)", self.scan_frequency_hz, self.points_per_scan,
                    self.min_distance_m, self.max_distance_m)

    def start(self) -> None:
        self._scanning = True
        logger.info("MockLidar scanning")

    def stop(self) -> None:
        self._scanning = False

    def shutdown(self) -> None:
        self.stop()
        self._connected = False

    # -- scanning -----------------------------------------------------

    def get_scan(self, timeout_s: float = 1.0) -> Optional[LidarScanPacket]:
        """Paced blocking read: sleeps one scan period, then generates a
        scan — the same shape of call the real driver makes into the SDK,
        so lidar_process's threading behaves identically either way.
        """
        if not self._scanning:
            return None

        period_s = 1.0 / self.scan_frequency_hz
        time.sleep(min(period_s, max(timeout_s, 0.0)))

        if period_s > timeout_s:
            return None  # honor the caller's timeout rather than overrunning it

        packet = self._generate_packet(period_s)
        self._latest_scan = packet
        self._last_scan_time = packet.timestamp
        return packet

    def _generate_packet(self, scan_duration_s: float) -> LidarScanPacket:
        t = time.time() - self._t0
        obstacle_angle_deg = (t * self.obstacle_sweep_deg_per_s) % 360.0 - 180.0

        points: list[LidarPoint] = []
        step = 360.0 / self.points_per_scan
        now_ts = time.time()

        for i in range(self.points_per_scan):
            angle_deg = -180.0 + i * step
            distance = self.room_radius_m

            angular_delta = abs(((angle_deg - obstacle_angle_deg) + 180.0) % 360.0 - 180.0)
            if angular_delta <= self.obstacle_half_width_deg:
                distance = self.obstacle_distance_m

            distance = max(self.min_distance_m, min(distance, self.max_distance_m))
            points.append(LidarPoint(
                angle_deg=angle_deg, distance_m=distance, intensity=1.0, timestamp=now_ts,
            ))

        self._sequence_id += 1
        return LidarScanPacket(
            points=points,
            timestamp=now_ts,
            sequence_id=self._sequence_id,
            frame_id=self.frame_id,
            source="mock_lidar",
            scan_duration_s=scan_duration_s,
            scan_frequency_hz=self.scan_frequency_hz,
            valid_points=len(points),
            invalid_points=0,
        )

    def get_latest_scan(self) -> Optional[LidarScanPacket]:
        return self._latest_scan

    def is_healthy(self) -> bool:
        if not self._scanning or self._last_scan_time == 0.0:
            return False
        max_gap_s = max(3.0 / self.scan_frequency_hz, 1.0)
        return (time.time() - self._last_scan_time) <= max_gap_s

    def get_status(self) -> LidarStatus:
        return LidarStatus(
            connected=self._connected,
            scanning=self._scanning,
            healthy=self.is_healthy(),
            port=None,
            baudrate=0,
            model=self.model,
            last_scan_timestamp=self._last_scan_time or None,
            # For the mock these coincide by construction: it generates at
            # exactly the rate it was told to. On real hardware they can
            # differ, which is the whole point of reporting both.
            scan_rate_hz=self.scan_frequency_hz if self._scanning else 0.0,
            expected_scan_rate_hz=self.scan_frequency_hz,
            error_message=None,
            simulated=True,
        )


def build_mock_lidar_from_config(cfg: dict) -> MockLidar:
    l_cfg = cfg["lidar"]
    sim_cfg = l_cfg.get("simulation", {})
    return MockLidar(
        # The mock actually generates at this rate, so for simulation the
        # configured expectation and the real rate are the same number.
        scan_frequency_hz=l_cfg.get("expected_scan_frequency_hz", 10.0),
        min_distance_m=l_cfg.get("min_distance", l_cfg.get("min_distance_m", 0.12)),
        max_distance_m=l_cfg.get("max_distance", l_cfg.get("max_distance_m", 10.0)),
        frame_id=l_cfg.get("frame_id", "lidar"),
        points_per_scan=sim_cfg.get("points_per_scan", 360),
        room_radius_m=sim_cfg.get("room_radius_m", 4.0),
        obstacle_distance_m=sim_cfg.get("obstacle_distance_m", 0.6),
        obstacle_half_width_deg=sim_cfg.get("obstacle_half_width_deg", 12.0),
        obstacle_sweep_deg_per_s=sim_cfg.get("obstacle_sweep_deg_per_s", 20.0),
    )
