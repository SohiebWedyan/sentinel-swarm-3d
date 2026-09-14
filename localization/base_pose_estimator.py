"""Unified pose estimation interface.

Outdoor mode fuses GPS + IMU + wheel encoders (+ optional visual odometry);
indoor mode fuses visual odometry/SLAM + IMU + wheel encoders. Both modes
implement this same interface, so navigation/mapping/swarm code calls
get_pose() without caring which sensors are behind it — that decision is
made once, in config/system.yaml + config/robots.yaml.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from common.types import Pose3D


class PoseEstimator(ABC):
    @abstractmethod
    def update(self, dt: float) -> None:
        """Advance the filter by dt seconds, consuming whatever sensor
        inputs have arrived since the last call (GPS fix, IMU sample,
        encoder ticks, VO/SLAM pose — implementation-specific).
        """

    @abstractmethod
    def get_pose(self) -> Pose3D:
        """Current best pose estimate, in the world/local frame."""

    @abstractmethod
    def get_velocity(self) -> tuple[float, float, float]:
        """Current linear velocity estimate (vx, vy, vz), world frame, m/s."""

    @abstractmethod
    def get_covariance(self):
        """Pose covariance (6x6 numpy array: x,y,z,roll,pitch,yaw), for
        confidence-aware fusion in the semantic map and formation control.
        """
