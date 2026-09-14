"""Perception visualization panel (Phase 2).

Its own module alongside dashboard.py and lidar_view.py, per the modular
visualization rule — each panel takes plain published data and returns an
image, holds no live reference into a processing thread, and can never
block or slow the pipeline it is displaying. dashboard.py (used by main.py)
is untouched.

Honest labelling is a hard requirement here: the depth panel states the
depth kind and normalizes for display only. A relative depth map is
rendered with a colour ramp and labelled RELATIVE — no distance numbers are
printed next to a detection, because monocular depth does not know
distances.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from common.types import CameraStatus, DepthKind, PerceptionHealth, PerceptionResult

logger = logging.getLogger(__name__)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False

_WINDOW_NAME = "SentinelSwarm-3D | Perception"


class PerceptionView:
    def __init__(self, show_depth: bool = True, window_scale: float = 1.0) -> None:
        self.show_depth = show_depth
        self.window_scale = window_scale
        self._gui_available = _HAS_CV2
        self._warned_no_gui = False

    def render(
        self,
        result: Optional[PerceptionResult],
        health: Optional[PerceptionHealth],
        camera_status: Optional[CameraStatus],
    ) -> Optional[np.ndarray]:
        if not _HAS_CV2:
            return None

        canvas = self._base_canvas(result)
        if result is not None:
            self._draw_detections(canvas, result)

        if self.show_depth and result is not None and result.depth is not None:
            depth_panel = self._render_depth(result, canvas.shape[0])
            if depth_panel is not None:
                canvas = np.hstack([canvas, depth_panel])

        self._draw_hud(canvas, result, health, camera_status)

        if self.window_scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.window_scale, fy=self.window_scale)

        self._try_show(canvas)
        return canvas

    def _base_canvas(self, result: Optional[PerceptionResult]) -> np.ndarray:
        if result is not None and result.image is not None:
            return result.image.copy()

        # No frame published (publish_frame: false, or nothing captured yet) —
        # still render the HUD so the operator sees WHY the picture is missing.
        height = max(result.image_height, 360) if result else 360
        width = max(result.image_width, 640) if result else 640
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(canvas, "no frame published", (20, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2, cv2.LINE_AA)
        return canvas

    def _draw_detections(self, canvas: np.ndarray, result: PerceptionResult) -> None:
        for det in result.detections:
            x1, y1, x2, y2 = (int(v) for v in (det.x1, det.y1, det.x2, det.y2))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 60), 2)

            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1),
                          (60, 220, 60), -1)
            cv2.putText(canvas, label, (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _render_depth(self, result: PerceptionResult, target_height: int) -> Optional[np.ndarray]:
        depth_frame = result.depth
        if depth_frame is None or depth_frame.depth is None:
            return None

        depth = depth_frame.depth
        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            norm = np.zeros(depth.shape, dtype=np.uint8)
        else:
            d_min, d_max = float(finite.min()), float(finite.max())
            span = max(d_max - d_min, 1e-6)
            # Display normalization ONLY. This rescaling is what makes the
            # map viewable; it is not a calibration and recovers no units.
            norm = np.clip((depth - d_min) / span * 255.0, 0, 255).astype(np.uint8)

        colored = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
        target_width = int(colored.shape[1] * (target_height / colored.shape[0]))
        colored = cv2.resize(colored, (max(target_width, 1), target_height))

        kind_text = depth_frame.kind.value.upper()
        if depth_frame.kind is not DepthKind.METRIC:
            kind_text += " (not meters)"
        if depth_frame.simulated:
            kind_text += " [SIMULATED]"

        cv2.putText(colored, f"DEPTH: {kind_text}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (60, 60, 220) if depth_frame.simulated else (240, 240, 240),
                    1, cv2.LINE_AA)
        cv2.putText(colored, "colour = display normalization, not distance", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        return colored

    def _draw_hud(
        self,
        canvas: np.ndarray,
        result: Optional[PerceptionResult],
        health: Optional[PerceptionHealth],
        camera_status: Optional[CameraStatus],
    ) -> None:
        y = 24
        if camera_status is not None:
            color = (60, 220, 60) if camera_status.healthy else (60, 60, 220)
            tag = "SIM" if camera_status.simulated else "CAM"
            cv2.putText(canvas,
                        f"[{tag}] {camera_status.device} {camera_status.width}x"
                        f"{camera_status.height} {camera_status.measured_fps:.1f}fps "
                        f"dropped={camera_status.frames_dropped}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y += 20

        if health is not None:
            color = (60, 220, 60) if health.healthy else (60, 60, 220)
            state = "OK" if health.healthy else "DEGRADED"
            cv2.putText(canvas,
                        f"PERCEPTION {state} | {health.fps:.1f} FPS | "
                        f"processed={health.frames_processed} stale={health.frames_dropped_stale}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y += 20
            if health.detector_degraded:
                cv2.putText(canvas, "YOLO DEGRADED - no model loaded, zero detections",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 220), 1, cv2.LINE_AA)
                y += 20

        if result is not None:
            cv2.putText(canvas,
                        f"frame #{result.sequence_id} | {result.detection_count} det | "
                        f"yolo {result.timing.detector_ms:.0f}ms  depth "
                        f"{result.timing.depth_ms:.0f}ms  total "
                        f"{result.timing.total_ms:.0f}ms  latency "
                        f"{result.timing.pipeline_latency_ms:.0f}ms",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    def _try_show(self, canvas: np.ndarray) -> None:
        if not self._gui_available:
            return
        try:
            cv2.imshow(_WINDOW_NAME, canvas)
            cv2.waitKey(1)
        except cv2.error as exc:
            self._gui_available = False
            if not self._warned_no_gui:
                logger.warning("No display/GUI backend available for the perception view "
                               "(%s). Continuing headless.", exc)
                self._warned_no_gui = True

    def close(self) -> None:
        if _HAS_CV2 and self._gui_available:
            try:
                cv2.destroyWindow(_WINDOW_NAME)
            except cv2.error:
                pass


def build_perception_view_from_config(cfg: dict) -> PerceptionView:
    v_cfg = cfg.get("perception", {}).get("visualization", {})
    return PerceptionView(
        show_depth=v_cfg.get("show_depth_map", True),
        window_scale=v_cfg.get("window_scale", 1.0),
    )
