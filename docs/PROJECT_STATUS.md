# SentinelSwarm-3D - Project Status

**Date:** 2026-09-14
**Scope:** Leader UGV only. Swarm operation is refused at runtime.
**Tests:** 188 passed, 3 skipped - measured on the Jetson Orin Nano.

## Executive summary

A modular perception and LiDAR runtime for a single Leader UGV. The suite now
runs **on the Jetson itself**, and one component - the RealSense D435 - has
committed hardware measurements.

The robot cannot move: no motor-control code exists, by design.

Two facts define the current state more than any other:

1. **The D435 is validated but not consumed.** Metric depth is measured and on
   file; the perception pipeline still runs on relative monocular depth.
2. **Nothing has run end-to-end on the robot.** Passing tests on the Jetson is
   not the same as a camera frame travelling through the pipeline to a
   decision.

## Status by layer

| # | Layer | Status |
|---|---|---|
| 1 | Hardware | PARTIALLY IMPLEMENTED - D435 **validated**, X4 Pro driver works standalone; no IMU, encoders, motors, GPS |
| 2 | Perception | PARTIALLY IMPLEMENTED - detector, depth, tracking, fusion |
| 3 | Geometry | CURRENT |
| 4 | Localization | PLANNED - interface only |
| 5 | Mapping | PARTIALLY IMPLEMENTED - semantic object DB; no occupancy grid, no point cloud |
| 6 | Navigation | PLANNED - interface only |
| 7 | Swarm communication | PLANNED - refused at runtime |
| 8 | Formation control | PLANNED - not started |
| 9 | Semantic reconstruction (ShapeR) | PLANNED - optional by design |
| 10 | Visualization | PARTIALLY IMPLEMENTED |

## What is genuinely working

- Leader-only process supervision with bounded restarts. ProcessManager
  **refuses** swarm processes rather than documenting that they are disabled.
- Bounded latest-value IPC - a slow consumer costs freshness, never memory.
- Camera capture with V4L2 tuning, FPS pacing, reconnect with backoff.
- Detection and depth on a **synchronized** frame, with decoupled cadence.
- DepthKind unit tagging - relative depth cannot masquerade as metric.
- YDLiDAR X4 Pro driver on the official SDK, with SDK-verified port detection.
- **D435 validated against the physical unit** - identity, firmware, USB mode,
  factory intrinsics.
- Loud, documented degradation everywhere; the pipeline runs with zero ML
  dependencies installed.
- 188 tests passing on the Jetson Orin Nano.

## What does not work, or does not exist

| Item | Status |
|---|---|
| D435 metric depth in the pipeline | **NOT WIRED IN** - grep finds no reference in runtime/ or perception/ |
| Metric 3D positions | **BLOCKED** - pipeline depth is RELATIVE |
| End-to-end run on the robot | **NEVER DONE** |
| Generic camera intrinsics | **ESTIMATE** - calibrate before trusting projection. The D435's are factory-measured and must NOT be calibrated |
| Camera-to-LiDAR extrinsics | **MISSING** |
| Occupancy grid | **DOES NOT EXIST** |
| Pose estimate | **DOES NOT EXIST** - the robot does not know where it is |
| SLAM | **DOES NOT EXIST** - no algorithm selected |
| Path planning | **DOES NOT EXIST** |
| Motor control | **DOES NOT EXIST - deliberately** |
| Swarm / followers | **REFUSED AT RUNTIME** |
| Literature review | **NOT CONDUCTED** |

## Measurements

Every measured number in this project, with its conditions.

| Measurement | Value | Conditions |
|---|---|---|
| **D435 colour intrinsics** | fx=605.03 fy=604.62 ppx=322.22 ppy=248.98 | 640x480, factory, serial 215322073372 |
| **D435 depth intrinsics** | fx=fy=386.19 ppx=326.11 ppy=238.56 | 640x480, factory |
| **D435 firmware / USB** | 5.12.7.150 / USB 3.2 | measured |
| **LiDAR scan rate** | **~4 Hz** | this X4 Pro unit, standalone. Below the 6-12 Hz band X4ProProfile validates against. The configured 10.0 exists to satisfy that range - **10 Hz is not claimed as achieved** |
| Test suite | 188 passed, 3 skipped | Jetson Orin Nano, 2026-09-14 |

No YOLO latency, no depth latency and no pipeline accuracy figure has been
measured on the Orin Nano.

## Open issues

| # | Issue | Severity |
|---|---|---|
| 1 | D435 metric depth not consumed by the pipeline | **Highest-value unstarted work** |
| 2 | No end-to-end run on the robot | **Blocks any real claim about the pipeline** |
| 3 | LiDAR full-runtime scan reception | **DEFERRED by instruction** - blocks SLAM |
| 4 | Measured ~4 Hz below the 6-12 Hz validation band | Open, uninvestigated |
| 5 | Generic camera intrinsics uncalibrated; no camera-to-LiDAR extrinsics | Open |
| 6 | Whether the chosen motors accept encoders | Open - **blocks a purchase decision** |

## Standing constraints

- Motors remain disabled. No motor code, no navigation commands, no motion.
- LiDAR scans are never faked; health is never forced true.
- Tests are never weakened to obtain a pass.
- Nothing is reported as implemented when it is planned or partial.
- ShapeR is never a real-time dependency.
