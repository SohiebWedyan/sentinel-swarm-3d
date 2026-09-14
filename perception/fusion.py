"""3D Object Fusion.

Combines one frame's 2D detections (from the YOLO fork) with the current
depth map (from the DA3 fork) into fully-populated DetectedObject3D
instances — this is the "3D Object Fusion" box in the spec's perception
pipeline diagram, sitting right before Object Tracking.

Depth extraction deliberately uses a median over a configurable central
sub-region of the bounding box rather than a single pixel: box edges very
often straddle background, so a single-pixel or full-box-mean sample is
noisy or biased toward the background depth.
"""

from __future__ import annotations

import logging

import numpy as np

from common.types import BBox, CameraIntrinsics, Detection2D, DetectedObject3D, Pose3D, new_object_id
from geometry.transforms import camera_to_world, pixel_to_camera_coords

logger = logging.getLogger(__name__)


class Object3DFusion:
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        camera_extrinsics: Pose3D,
        bbox_subregion_fraction: float = 0.4,
        min_valid_depth_m: float = 0.15,
        max_valid_depth_m: float = 40.0,
        min_valid_depth_samples: int = 5,
        source_robot: str = "unknown",
    ) -> None:
        self.intrinsics = intrinsics
        self.camera_extrinsics = camera_extrinsics
        self.bbox_subregion_fraction = bbox_subregion_fraction
        self.min_valid_depth_m = min_valid_depth_m
        self.max_valid_depth_m = max_valid_depth_m
        self.min_valid_depth_samples = min_valid_depth_samples
        self.source_robot = source_robot

    def fuse(
        self,
        detections: list[Detection2D],
        depth_map: np.ndarray | None,
        depth_confidence_map: np.ndarray | None,
        image_shape: tuple[int, int],
        robot_pose_world: Pose3D | None = None,
    ) -> list[DetectedObject3D]:
        """Turn one frame's detections + depth map into DetectedObject3D
        instances. Returns an empty list (never raises) if depth is not yet
        available, e.g. the async depth worker hasn't produced a first
        result — callers should treat that as "no 3D info this frame" and
        keep going, not as a fatal error.
        """
        if depth_map is None:
            return []

        objects: list[DetectedObject3D] = []
        img_h, img_w = image_shape[:2]
        depth_h, depth_w = depth_map.shape[:2]

        for det in detections:
            median_depth, sample_count = self._median_depth_in_bbox(
                det.bbox, depth_map, img_w, img_h, depth_w, depth_h
            )
            if median_depth is None or sample_count < self.min_valid_depth_samples:
                logger.debug("Skipping detection %s: insufficient valid depth samples "
                             "(%s)", det.class_name, sample_count)
                continue

            if not (self.min_valid_depth_m <= median_depth <= self.max_valid_depth_m):
                continue

            u, v = det.bbox.center
            u_depth = u * (depth_w / img_w)
            v_depth = v * (depth_h / img_h)
            camera_coords = pixel_to_camera_coords(u_depth, v_depth, median_depth, self.intrinsics)

            world_coords = camera_to_world(camera_coords, self.camera_extrinsics, robot_pose_world)

            dimensions = self._estimate_dimensions(det.bbox, median_depth)

            depth_conf = 0.5
            if depth_confidence_map is not None:
                depth_conf = self._sample_confidence(det.bbox, depth_confidence_map, img_w, img_h, depth_w, depth_h)

            objects.append(
                DetectedObject3D(
                    object_id=new_object_id(),
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    camera_coords=camera_coords,
                    world_coords=world_coords,
                    dimensions=dimensions,
                    bbox=det.bbox,
                    depth_confidence=depth_conf,
                    timestamp=det.frame_timestamp,
                    source_robot=self.source_robot,
                )
            )

        return objects

    def _median_depth_in_bbox(
        self,
        bbox: BBox,
        depth_map: np.ndarray,
        img_w: int,
        img_h: int,
        depth_w: int,
        depth_h: int,
    ) -> tuple[float | None, int]:
        """Median depth over a configurable central fraction of the box,
        mapped into the (possibly lower-resolution) depth map's pixel grid.
        """
        scale_x = depth_w / img_w
        scale_y = depth_h / img_h

        cx, cy = bbox.center
        half_w = (bbox.width * self.bbox_subregion_fraction / 2.0)
        half_h = (bbox.height * self.bbox_subregion_fraction / 2.0)

        x1 = int(max((cx - half_w) * scale_x, 0))
        x2 = int(min((cx + half_w) * scale_x, depth_w - 1))
        y1 = int(max((cy - half_h) * scale_y, 0))
        y2 = int(min((cy + half_h) * scale_y, depth_h - 1))

        if x2 <= x1 or y2 <= y1:
            return None, 0

        region = depth_map[y1:y2, x1:x2]
        valid = region[np.isfinite(region) & (region > 0)]
        if valid.size == 0:
            return None, 0

        return float(np.median(valid)), int(valid.size)

    def _sample_confidence(
        self, bbox: BBox, conf_map: np.ndarray, img_w: int, img_h: int, depth_w: int, depth_h: int
    ) -> float:
        scale_x, scale_y = depth_w / img_w, depth_h / img_h
        cx, cy = bbox.center
        px = int(min(max(cx * scale_x, 0), depth_w - 1))
        py = int(min(max(cy * scale_y, 0), depth_h - 1))
        return float(conf_map[py, px])

    def _estimate_dimensions(self, bbox: BBox, depth_m: float) -> tuple[float, float, float]:
        """Rough real-world (width, height, depth) from bbox size + range,
        using the pinhole similar-triangles relationship. This is a coarse
        Phase-1 estimate — good enough for formation/obstacle sizing, not
        for precise metrology.
        """
        real_w = bbox.width * depth_m / self.intrinsics.fx
        real_h = bbox.height * depth_m / self.intrinsics.fy
        assumed_depth_extent = (real_w + real_h) / 2.0  # no multi-view shape info yet
        return (real_w, real_h, assumed_depth_extent)


def build_fusion_from_config(cfg: dict, source_robot: str = "unknown") -> Object3DFusion:
    intr_cfg = cfg["camera"]["intrinsics"]
    intrinsics = CameraIntrinsics(
        fx=intr_cfg["fx"], fy=intr_cfg["fy"], cx=intr_cfg["cx"], cy=intr_cfg["cy"],
        distortion=tuple(intr_cfg.get("distortion", [0, 0, 0, 0, 0])),
    )

    extr_cfg = cfg["camera"].get("extrinsics", {})
    translation = extr_cfg.get("translation", [0.0, 0.0, 0.0])
    rotation = extr_cfg.get("rotation_rpy", [0.0, 0.0, 0.0])
    camera_extrinsics = Pose3D(
        x=translation[0], y=translation[1], z=translation[2],
        roll=rotation[0], pitch=rotation[1], yaw=rotation[2],
        frame="robot",
    )

    f_cfg = cfg["perception"]["fusion"]
    return Object3DFusion(
        intrinsics=intrinsics,
        camera_extrinsics=camera_extrinsics,
        bbox_subregion_fraction=f_cfg.get("bbox_subregion_fraction", 0.4),
        min_valid_depth_m=f_cfg.get("min_valid_depth_m", 0.15),
        max_valid_depth_m=f_cfg.get("max_valid_depth_m", 40.0),
        min_valid_depth_samples=f_cfg.get("min_valid_depth_samples", 5),
        source_robot=source_robot,
    )
