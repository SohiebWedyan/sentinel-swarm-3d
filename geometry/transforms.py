"""Coordinate transform utilities.

Implements exactly the pixel -> camera -> world chain from the spec:

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth(u, v)

    P_world = T_world_camera * P_camera

Phase 1 only needs pixel_to_camera_coords() (the fusion module) plus a
camera->world transform that is the identity by default (config/camera.yaml
extrinsics + a Pose3D from later phases). The 4x4 homogeneous transform
helpers here are what localization/mapping build on starting Phase 3, so
this module is written once and reused rather than re-derived per layer.
"""

from __future__ import annotations

import numpy as np

from common.types import CameraIntrinsics, Pose3D


def pixel_to_camera_coords(u: float, v: float, z: float, intrinsics: CameraIntrinsics) -> tuple[float, float, float]:
    """Back-project a pixel + depth to a 3D point in the camera frame.

    (u, v) pixel coordinates, z depth in meters (camera Z axis, forward).
    Returns (X, Y, Z) in meters, camera frame (X right, Y down, Z forward —
    standard OpenCV pinhole convention).
    """
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return (float(x), float(y), float(z))


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Standard Z-Y-X (yaw-pitch-roll) rotation matrix, radians in, 3x3 out."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    r_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    r_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    r_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return r_z @ r_y @ r_x


def pose_to_homogeneous(pose: Pose3D) -> np.ndarray:
    """Pose3D -> 4x4 homogeneous transform matrix (this pose's frame -> parent frame)."""
    t_mat = np.eye(4)
    t_mat[:3, :3] = rpy_to_rotation_matrix(pose.roll, pose.pitch, pose.yaw)
    t_mat[:3, 3] = [pose.x, pose.y, pose.z]
    return t_mat


def transform_point(point_xyz: tuple[float, float, float], transform: np.ndarray) -> tuple[float, float, float]:
    """Apply a 4x4 homogeneous transform to a 3D point."""
    p = np.array([*point_xyz, 1.0])
    result = transform @ p
    return (float(result[0]), float(result[1]), float(result[2]))


def camera_to_world(
    point_camera: tuple[float, float, float],
    camera_extrinsics: Pose3D,
    robot_pose_world: Pose3D | None = None,
) -> tuple[float, float, float]:
    """P_world = T_world_robot * T_robot_camera * P_camera.

    Phase 1 has no localization yet, so robot_pose_world defaults to the
    world origin (identity) — this is exactly the "T_world_camera" from the
    spec collapsing to the static camera extrinsics until Phase 3 supplies a
    real PoseEstimator.
    """
    t_robot_camera = pose_to_homogeneous(camera_extrinsics)
    point_robot = transform_point(point_camera, t_robot_camera)

    if robot_pose_world is None:
        return point_robot

    t_world_robot = pose_to_homogeneous(robot_pose_world)
    return transform_point(point_robot, t_world_robot)
