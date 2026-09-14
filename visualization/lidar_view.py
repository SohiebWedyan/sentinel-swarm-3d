"""LiDAR live-scan visualization panel.

A separate module from visualization/dashboard.py (which is camera/YOLO/
depth-specific and is not touched by this file) — per the "modular
visualization architecture" requirement, each panel is its own small
renderer that takes plain data and returns an image, never a live
reference into the LiDAR process. Headless-safe exactly like dashboard.py:
if no GUI backend is available it logs once and skips cv2.imshow(), while
still returning the composed frame.

Visualization is strictly read-only and never on the safety path: it
consumes already-published values (packet, sectors, status) and cannot
block or slow acquisition.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from common.types import (
    LidarSafetyLevel,
    LidarScanPacket,
    LidarSectorName,
    LidarStatus,
    SectorClearance,
)

logger = logging.getLogger(__name__)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False

_WINDOW_NAME = "SentinelSwarm-3D | LiDAR"

_SAFETY_COLORS = {
    LidarSafetyLevel.NORMAL: (60, 220, 60),
    LidarSafetyLevel.CAUTION: (60, 200, 240),
    LidarSafetyLevel.STOP: (60, 120, 240),
    LidarSafetyLevel.EMERGENCY: (60, 60, 240),
    LidarSafetyLevel.NO_DATA: (150, 150, 150),
}


class LidarView:
    def __init__(self, canvas_size: int = 640, max_range_m: float = 10.0) -> None:
        self.canvas_size = canvas_size
        self.max_range_m = max_range_m
        self._gui_available = _HAS_CV2
        self._warned_no_gui = False

    def render(
        self,
        scan: Optional[LidarScanPacket],
        sectors: Optional[dict[LidarSectorName, SectorClearance]],
        status: Optional[LidarStatus],
        safety_level: LidarSafetyLevel = LidarSafetyLevel.NO_DATA,
    ) -> Optional[np.ndarray]:
        if not _HAS_CV2:
            return None

        size = self.canvas_size
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        center = size // 2
        scale = (size / 2 - 20) / self.max_range_m

        for r in range(2, int(self.max_range_m) + 1, 2):
            cv2.circle(canvas, (center, center), int(r * scale), (40, 40, 40), 1)
            cv2.putText(canvas, f"{r}m", (center + int(r * scale) - 22, center - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1, cv2.LINE_AA)

        if scan is not None:
            for point in scan.points:
                if point.distance_m <= 0:
                    continue
                theta = np.radians(point.angle_deg)
                x = center + int(point.distance_m * scale * np.sin(theta))
                y = center - int(point.distance_m * scale * np.cos(theta))
                if 0 <= x < size and 0 <= y < size:
                    cv2.circle(canvas, (x, y), 2, (60, 220, 60), -1)

        cv2.drawMarker(canvas, (center, center), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)

        self._draw_status(canvas, scan, status, safety_level)
        self._draw_sectors(canvas, sectors)

        self._try_show(canvas)
        return canvas

    def _draw_status(
        self,
        canvas: np.ndarray,
        scan: Optional[LidarScanPacket],
        status: Optional[LidarStatus],
        safety_level: LidarSafetyLevel,
    ) -> None:
        if status is None:
            cv2.putText(canvas, "LiDAR: NO STATUS", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (60, 60, 220), 2, cv2.LINE_AA)
            return

        health_color = (60, 220, 60) if status.healthy else (60, 60, 220)
        health_text = "OK" if status.healthy else "UNHEALTHY"
        mode_text = "SIM" if status.simulated else "HW"

        cv2.putText(canvas, f"LiDAR [{mode_text}] {status.model}: {health_text}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, health_color, 2, cv2.LINE_AA)

        port_text = f"port={status.port or 'n/a'} baud={status.baudrate or 'n/a'} " \
                    f"rate={status.scan_rate_hz:.1f}/{status.expected_scan_rate_hz:.1f}Hz"
        cv2.putText(canvas, port_text, (10, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (190, 190, 190), 1, cv2.LINE_AA)

        if scan is not None:
            scan_text = f"scan #{scan.sequence_id}  valid={scan.valid_points} " \
                        f"rejected={scan.invalid_points}"
            cv2.putText(canvas, scan_text, (10, 62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (190, 190, 190), 1, cv2.LINE_AA)

        safety_color = _SAFETY_COLORS.get(safety_level, (150, 150, 150))
        cv2.putText(canvas, f"FRONT SAFETY: {safety_level.value.upper()}",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, safety_color, 2, cv2.LINE_AA)

        if status.error_message:
            cv2.putText(canvas, status.error_message[:70], (10, self.canvas_size - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 220), 1, cv2.LINE_AA)

    def _draw_sectors(
        self, canvas: np.ndarray, sectors: Optional[dict[LidarSectorName, SectorClearance]]
    ) -> None:
        if not sectors:
            return
        y0 = 108
        for name, clearance in sectors.items():
            min_d = clearance.min_distance_m
            min_d_text = f"{min_d:.2f}m" if min_d is not None else "n/a"
            obstacle_text = " OBSTACLE" if clearance.obstacle_detected else ""
            label = f"{name.value:<12} min={min_d_text}{obstacle_text}"
            color = (60, 60, 220) if clearance.obstacle_detected else (200, 200, 200)
            cv2.putText(canvas, label, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y0 += 18

    def _try_show(self, canvas: np.ndarray) -> None:
        if not self._gui_available:
            return
        try:
            cv2.imshow(_WINDOW_NAME, canvas)
            cv2.waitKey(1)
        except cv2.error as exc:
            self._gui_available = False
            if not self._warned_no_gui:
                logger.warning("No display/GUI backend available for the LiDAR view (%s). "
                               "Continuing headless.", exc)
                self._warned_no_gui = True

    def close(self) -> None:
        if _HAS_CV2 and self._gui_available:
            try:
                cv2.destroyWindow(_WINDOW_NAME)
            except cv2.error:
                pass
