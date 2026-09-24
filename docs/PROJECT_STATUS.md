# SentinelSwarm-3D - Project Status

**Date:** 2026-09-23
**Scope:** Leader UGV only. Swarm operation is refused at runtime.

## Executive summary

A single Leader UGV with two software paths:

1. **ROS 2 Humble navigation stack** (`ros2_bringup/`): RPLIDAR C1, Intel
   RealSense D435, ICP odometry, RTAB-Map SLAM and Nav2. This is the path
   the robot's navigation is being built on.
2. **Python perception / leader runtime** (repository root): process
   supervision, camera capture, YOLO detection, monocular depth, 3D object
   fusion and tracking.

What defines the current state:

- **The sensing, SLAM and Nav2 pipeline is validated on the Jetson** up to
  velocity commands: LiDAR scan, ICP odometry, RTAB-Map, D435 obstacles in
  the local costmap, global planning, and controller output on `/cmd_vel`.
- **The robot has not yet driven autonomously.** Physical motor/chassis
  integration is not complete, so no end-to-end physical navigation has
  been validated.
- **No clean, final room map exists yet.** Only partial test maps have
  been produced.
- **The validated controller behaviour depends on an external patch.** A
  Nav2 controller goal-check issue was addressed with a patched
  `nav2_controller` built in a separate overlay workspace. That patch is
  **not** part of this repository.

## Hardware

| Component | Status |
|---|---|
| NVIDIA Jetson Orin Nano 8 GB | Main compute. Runs ROS 2, SLAM, Nav2 and perception |
| Raspberry Pi 4 | Low-level hardware / control. Integration in progress |
| RPLIDAR C1 | **Validated.** `/dev/ttyUSB0`, 460800 baud, `rplidar_ros` |
| Intel RealSense D435 | **Validated.** RGB + depth, PointCloud2. No internal IMU (not a D435i) |
| External IMU | Present on the platform; not used by the current ROS 2 launch |
| 2WD differential-drive chassis | Motor/chassis integration **not complete** |

## Status by layer

| # | Layer | Status |
|---|---|---|
| 1 | Hardware | PARTIALLY INTEGRATED - LiDAR and D435 validated in ROS 2; motor/chassis integration not complete |
| 2 | Perception (Python) | PARTIALLY IMPLEMENTED - detector, monocular depth, tracking, fusion |
| 3 | Geometry | CURRENT |
| 4 | Odometry / localization | **VALIDATED (ROS 2)** - ICP odometry from the LiDAR. Python `localization/` is interface-only |
| 5 | Mapping | **SLAM VALIDATED (ROS 2)** - RTAB-Map, LiDAR-only. Clean final room map **not completed**. Python semantic object DB |
| 6 | Navigation | **PARTIALLY VALIDATED (ROS 2)** - Nav2 planning, costmaps and `/cmd_vel` output validated; physical driving not validated |
| 7 | Swarm communication | PLANNED - refused at runtime |
| 8 | Formation control | PLANNED - not started |
| 9 | Semantic reconstruction (ShapeR) | PLANNED - optional by design |
| 10 | Visualization | PARTIALLY IMPLEMENTED - RViz2 lightweight config; Python dashboards |

## ROS 2 stack

| Item | Configuration |
|---|---|
| ROS distribution | ROS 2 Humble |
| LiDAR | RPLIDAR C1, `/dev/ttyUSB0`, 460800 baud, `/scan` |
| Static TF | `base_link → laser`, `base_link → camera_link` |
| Odometry | ICP odometry (`rtabmap_odom`), `/odom`, `odom → base_link` |
| SLAM | RTAB-Map, LiDAR-only, `map → odom` and the occupancy map |
| Localization in Nav2 | None separate: no AMCL, no `map_server`; RTAB-Map provides both |
| D435 point cloud | Enabled with `pointcloud__neon_.enable` (ARM-specific parameter) |
| D435 topic | `/camera/camera/depth/color/points` (PointCloud2) |
| Global costmap | LiDAR only (`/scan`): static + obstacle + inflation layers |
| Local costmap | Voxel layer with two sources: LiDAR `/scan` and D435 PointCloud2 |
| Visualization | RViz2 with `ros2_bringup/rviz/sentinel_lite.rviz` |

The D435 is deliberately **not** used for SLAM and **not** used in the
global costmap. It contributes obstacles above and below the LiDAR scan
plane to the local costmap only.

`sentinel_lite.rviz` displays TF, LaserScan, Map, Global Costmap, Local
Costmap, Global Plan, Local Plan, Footprint, Odometry and Grid. It avoids
PointCloud2 and Image displays on purpose, because rendering the D435
cloud in RViz on the Orin Nano is heavy enough to freeze the session.

### External Nav2 controller patch

The validated Nav2 controller behaviour depends on a patched
`nav2_controller` that addresses a goal-check issue. The patch is built in
a separate overlay workspace, **outside this repository**, and is sourced
on top of ROS 2 Humble at runtime. This repository contains **no copy of
that patch**. A checkout of this repository running the stock Humble
`nav2_controller` may behave differently at the goal.

## Validation summary

| Item | Status |
|---|---|
| RPLIDAR C1 `/scan` | **Validated** |
| ICP odometry | **Validated** |
| RTAB-Map SLAM | **Validated** |
| D435 RGB + depth | **Validated** |
| D435 PointCloud2 integrated into the Nav2 local voxel costmap | **Validated** |
| Local costmap with LiDAR + D435 obstacle sources | **Validated** |
| Nav2 global planning | **Validated** |
| Nav2 controller velocity output on `/cmd_vel` | **Validated** (with the external controller patch) |
| Clean, final room map | **NOT COMPLETED** - only partial maps so far |
| Final autonomous mapping | **NOT COMPLETED** |
| Physical autonomous driving | **NOT VALIDATED** - motor/chassis integration not complete |
| Swarm / follower functionality | **NOT IMPLEMENTED** |

A map shown in RViz is a live SLAM view, not a finished map.

## Python runtime - what works

- Leader-only process supervision with bounded restarts. ProcessManager
  **refuses** swarm processes rather than documenting that they are disabled.
- Bounded latest-value IPC - a slow consumer costs freshness, never memory.
- Camera capture with V4L2 tuning, FPS pacing, reconnect with backoff.
- Detection and depth on a **synchronized** frame, with decoupled cadence.
- DepthKind unit tagging - relative depth cannot masquerade as metric.
- Standalone D435 validation tool (`tools/validate_realsense_d435.py`).
- Loud, documented degradation everywhere; the pipeline runs with zero ML
  dependencies installed.

The Python runtime's real-hardware LiDAR driver targets the YDLiDAR X4
Pro, the LiDAR used earlier in the project. The current RPLIDAR C1 is
driven through ROS 2, not through the Python driver.

## What does not work, or does not exist

| Item | Status |
|---|---|
| Physical autonomous driving | **NOT VALIDATED** - motor/chassis integration not complete |
| `/cmd_vel` to motors | **NOT CONNECTED** - no motor-control code in this repository |
| Clean final room map | **NOT COMPLETED** |
| Nav2 controller patch in this repository | **NOT PRESENT** - lives in an external overlay |
| Chassis measurements | **PLACEHOLDERS** - robot footprint, velocity/acceleration limits, LiDAR height and camera mount offsets must be measured on the final chassis |
| IMU in the ROS 2 stack | **NOT USED** - odometry is LiDAR-only (ICP) |
| D435 metric depth in the Python perception pipeline | **NOT WIRED IN** - the Python pipeline uses relative monocular depth |
| Metric 3D object positions (Python pipeline) | **BLOCKED** - pipeline depth is RELATIVE |
| Generic camera intrinsics | **ESTIMATE** - calibrate before trusting projection. The D435's are factory-measured and must NOT be calibrated |
| Swarm / followers | **NOT IMPLEMENTED - refused at runtime** |

## Measurements

| Measurement | Value | Conditions |
|---|---|---|
| **D435 colour intrinsics** | fx=605.03 fy=604.62 ppx=322.22 ppy=248.98 | 640x480, factory calibration |
| **D435 depth intrinsics** | fx=fy=386.19 ppx=326.11 ppy=238.56 | 640x480, factory calibration |
| **RPLIDAR C1 link** | `/dev/ttyUSB0`, 460800 baud | ROS 2 Humble, `rplidar_ros` |
| YDLiDAR X4 Pro scan rate (earlier LiDAR) | ~4 Hz | Standalone, Python driver. Historical; not the current LiDAR |

No YOLO latency, no depth-model latency and no pipeline accuracy figure has
been measured on the Orin Nano.

## Open issues

| # | Issue | Notes |
|---|---|---|
| 1 | Motor/chassis integration (Raspberry Pi 4, 2WD base) | Blocks physical autonomous driving |
| 2 | Clean, final room map | Not yet produced |
| 3 | Nav2 controller patch lives outside the repository | Validated behaviour is not reproducible from this repository alone |
| 4 | Chassis placeholders (footprint, limits, sensor mounts) | Must be measured on the final chassis |
| 5 | External IMU not yet used in odometry | Open |
| 6 | D435 metric depth not consumed by the Python perception pipeline | Open |

## Standing constraints

- Motors remain disabled (`motors_enabled: false` in `config/leader.yaml`).
- No motor-control code in this repository until the motor/chassis
  integration is designed; `/cmd_vel` is not yet connected to the motors.
- LiDAR scans are never faked; health is never forced true.
- Tests are never weakened to obtain a pass.
- Nothing is reported as implemented when it is planned or partial.
- ShapeR is never a real-time dependency.
