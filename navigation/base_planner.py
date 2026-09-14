"""Global/local planner interfaces.

GlobalPlanner: D* Lite today (config: navigation.global_planner.backend),
chosen because it natively supports incremental replanning on map updates
without a full from-scratch search — important once the occupancy map is
changing live from moving obstacles.

LocalPlanner: Dynamic Window Approach today, swappable for TEB/MPC/RL later
purely through navigation.local_planner.backend — nothing outside this
module should import DWA-specific types.

Obstacle input comes from the LiDAR (YDLiDAR X4 Pro) via
hardware/lidar/obstacle_sectors.py. The active Leader runtime has no
ultrasonic sensors; config/robots.yaml's follower entries still list them
for a future phase, but nothing in the safety or navigation path depends
on them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from common.types import Pose3D


class GlobalPlanner(ABC):
    @abstractmethod
    def set_goal(self, goal: Pose3D) -> None:
        """Set/replace the current navigation goal."""

    @abstractmethod
    def update_map(self, occupancy_map) -> None:
        """Incorporate a new/changed occupancy map. Implementations that
        support incremental replanning (D* Lite) should only recompute the
        affected region rather than replanning from scratch.
        """

    @abstractmethod
    def plan(self, current_pose: Pose3D) -> list[Pose3D]:
        """Return a global path (waypoints) from current_pose to the goal."""


class LocalPlanner(ABC):
    @abstractmethod
    def compute_velocity_command(
        self,
        current_pose: Pose3D,
        global_path: list[Pose3D],
        dynamic_obstacles: list,
        lidar_clearances: dict,
    ) -> tuple[float, float]:
        """Return (linear_velocity_mps, angular_velocity_radps) for this
        control cycle, given the current pose, the global path to track,
        AI-perceived dynamic obstacles, and LiDAR clearance information.

        lidar_clearances is the
        {LidarSectorName: SectorClearance} mapping produced by
        hardware/lidar/obstacle_sectors.py — the Leader's primary ranging
        sensor is the YDLiDAR X4 Pro, not ultrasonics, so this is the
        fusion point for "depth geometry + LiDAR clearance + occupancy map
        + dynamic object positions".
        """
