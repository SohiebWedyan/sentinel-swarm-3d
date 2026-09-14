"""Phase 1 monitoring dashboard.

Renders three of the panels called out in the spec's full visualization
list — live camera feed, YOLO detections, depth map — plus track IDs and
per-frame perception health text. The remaining panels (robot/leader/
follower position, semantic objects, planned paths, occupancy map, full
health status) are natural additions once localization/navigation/swarm
(Phases 3-7) exist to supply that data; render_panel() is written so adding
them later is a new function beside these, not a rewrite.

Runs headless-safely: if OpenCV's GUI backend isn't available (e.g. a
server/container with no display), it logs a warning once and skips
cv2.imshow() rather than crashing, while still returning the composed
frame so a caller could choose to write it to disk or stream it instead.
"""

from __future__ import annotations

import logging

import numpy as np

from common.types import DetectedObject3D

logger = logging.getLogger(__name__)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


class Dashboard:
    def __init__(
        self,
        show_camera_feed: bool = True,
        show_detections: bool = True,
        show_depth_map: bool = True,
        show_track_ids: bool = True,
        window_scale: float = 1.0,
        window_name: str = "SentinelSwarm-3D | Phase 1",
    ) -> None:
        self.show_camera_feed = show_camera_feed
        self.show_detections = show_detections
        self.show_depth_map = show_depth_map
        self.show_track_ids = show_track_ids
        self.window_scale = window_scale
        self.window_name = window_name
        self._gui_available = _HAS_CV2
        self._warned_no_gui = False

    def render(
        self,
        frame: np.ndarray,
        objects: list[DetectedObject3D],
        depth_map: np.ndarray | None,
        fps: float,
        depth_is_mock: bool,
    ) -> np.ndarray:
        """Compose the dashboard frame and, if a GUI is available, display
        it. Always returns the composed BGR image so callers can also save
        it, stream it, or run this headless in tests.
        """
        if not _HAS_CV2:
            return frame

        canvas = frame.copy() if self.show_camera_feed else np.zeros_like(frame)

        if self.show_detections:
            canvas = self._draw_detections(canvas, objects)

        if self.show_depth_map and depth_map is not None:
            depth_panel = self._colorize_depth(depth_map, frame.shape[1], frame.shape[0])
            canvas = self._side_by_side(canvas, depth_panel)

        canvas = self._draw_hud(canvas, fps, len(objects), depth_is_mock)

        if self.window_scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.window_scale, fy=self.window_scale)

        self._try_show(canvas)
        return canvas

    def _draw_detections(self, canvas: np.ndarray, objects: list[DetectedObject3D]) -> np.ndarray:
        for obj in objects:
            if obj.bbox is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in obj.bbox.as_xyxy())
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 60), 2)

            label = f"{obj.class_name} {obj.confidence:.2f}"
            if obj.camera_coords:
                label += f" Z={obj.camera_coords[2]:.2f}m"
            if self.show_track_ids and obj.track_id is not None:
                label = f"#{obj.track_id} {label}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (60, 220, 60), -1)
            cv2.putText(canvas, label, (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return canvas

    def _colorize_depth(self, depth_map: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        finite = depth_map[np.isfinite(depth_map)]
        if finite.size == 0:
            norm = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            d_min, d_max = float(finite.min()), float(finite.max())
            span = max(d_max - d_min, 1e-6)
            norm = np.clip((depth_map - d_min) / span * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
        return cv2.resize(colored, (target_w, target_h))

    def _side_by_side(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.hstack([left, right])

    def _draw_hud(self, canvas: np.ndarray, fps: float, num_objects: int, depth_is_mock: bool) -> np.ndarray:
        depth_tag = "MOCK DEPTH" if depth_is_mock else "DA3"
        text = f"FPS: {fps:5.1f}  |  Objects: {num_objects:2d}  |  Depth: {depth_tag}"
        cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def _try_show(self, canvas: np.ndarray) -> None:
        if not self._gui_available:
            return
        try:
            cv2.imshow(self.window_name, canvas)
            cv2.waitKey(1)
        except cv2.error as exc:
            self._gui_available = False
            if not self._warned_no_gui:
                logger.warning("No display/GUI backend available for cv2.imshow (%s). "
                               "Continuing headless; dashboard frames are still returned "
                               "from render() for saving/streaming.", exc)
                self._warned_no_gui = True

    def close(self) -> None:
        if _HAS_CV2 and self._gui_available:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass


def build_dashboard_from_config(cfg: dict) -> Dashboard:
    v_cfg = cfg["perception"]["visualization"]
    return Dashboard(
        show_camera_feed=v_cfg.get("show_camera_feed", True),
        show_detections=v_cfg.get("show_detections", True),
        show_depth_map=v_cfg.get("show_depth_map", True),
        show_track_ids=v_cfg.get("show_track_ids", True),
        window_scale=v_cfg.get("window_scale", 1.0),
    )
