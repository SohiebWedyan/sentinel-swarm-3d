"""Unit tests for the pixel -> camera -> world math and the fusion module's
median-depth extraction. These don't touch YOLO/DA3/OpenCV at all — they
exercise geometry.transforms and perception.fusion directly with hand-built
inputs, so they run anywhere (no GPU, no camera, no model weights).
"""

from __future__ import annotations

import numpy as np
import pytest

from common.types import BBox, CameraIntrinsics, Detection2D, Pose3D
from geometry.transforms import camera_to_world, pixel_to_camera_coords
from perception.fusion import Object3DFusion


def test_pixel_to_camera_coords_matches_pinhole_formula():
    intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
    x, y, z = pixel_to_camera_coords(u=740.0, v=360.0, z=5.0, intrinsics=intrinsics)

    assert z == 5.0
    assert x == pytest.approx((740.0 - 640.0) * 5.0 / 1000.0)
    assert y == pytest.approx((360.0 - 360.0) * 5.0 / 1000.0)


def test_camera_to_world_identity_extrinsics_is_passthrough():
    identity_extrinsics = Pose3D(x=0, y=0, z=0, roll=0, pitch=0, yaw=0)
    point_camera = (1.0, 2.0, 3.0)

    world = camera_to_world(point_camera, identity_extrinsics, robot_pose_world=None)

    assert world == pytest.approx(point_camera)


def test_camera_to_world_applies_translation():
    extrinsics = Pose3D(x=0.5, y=0.0, z=0.2, roll=0, pitch=0, yaw=0)
    point_camera = (0.0, 0.0, 1.0)

    world = camera_to_world(point_camera, extrinsics, robot_pose_world=None)

    assert world == pytest.approx((0.5, 0.0, 1.2))


def _make_fusion() -> Object3DFusion:
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    extrinsics = Pose3D()
    return Object3DFusion(
        intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        bbox_subregion_fraction=0.5,
        min_valid_depth_m=0.1,
        max_valid_depth_m=20.0,
        min_valid_depth_samples=1,
        source_robot="test_robot",
    )


def test_fusion_extracts_median_depth_and_populates_object():
    fusion = _make_fusion()

    # A uniform 3-meter depth map (matches image resolution -> scale factor 1.0)
    depth_map = np.full((480, 640), 3.0, dtype=np.float32)
    detection = Detection2D(
        bbox=BBox(x1=280.0, y1=200.0, x2=360.0, y2=280.0),
        class_id=0,
        class_name="person",
        confidence=0.9,
    )

    objects = fusion.fuse(
        detections=[detection],
        depth_map=depth_map,
        depth_confidence_map=None,
        image_shape=(480, 640, 3),
    )

    assert len(objects) == 1
    obj = objects[0]
    assert obj.class_name == "person"
    assert obj.camera_coords[2] == pytest.approx(3.0)
    assert obj.source_robot == "test_robot"
    assert obj.world_coords is not None


def test_fusion_rejects_detection_outside_valid_depth_range():
    fusion = _make_fusion()
    depth_map = np.full((480, 640), 100.0, dtype=np.float32)  # beyond max_valid_depth_m=20
    detection = Detection2D(
        bbox=BBox(x1=280.0, y1=200.0, x2=360.0, y2=280.0),
        class_id=0,
        class_name="person",
        confidence=0.9,
    )

    objects = fusion.fuse(
        detections=[detection],
        depth_map=depth_map,
        depth_confidence_map=None,
        image_shape=(480, 640, 3),
    )

    assert objects == []


def test_fusion_returns_empty_when_depth_not_yet_available():
    fusion = _make_fusion()
    detection = Detection2D(
        bbox=BBox(x1=280.0, y1=200.0, x2=360.0, y2=280.0),
        class_id=0,
        class_name="person",
        confidence=0.9,
    )

    objects = fusion.fuse(
        detections=[detection],
        depth_map=None,
        depth_confidence_map=None,
        image_shape=(480, 640, 3),
    )

    assert objects == []
