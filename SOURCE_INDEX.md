# Source Index

A map of this repository for a developer or AI agent arriving without context.
**Read this before changing anything.**

## Read these first, in this order

| # | File | Why |
|---|---|---|
| 1 | [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | What actually works today, and what does not |
| 2 | [README.md](README.md) | Architecture, hardware, how to run |
| 3 | [docs/hardware/realsense-d435.md](docs/hardware/realsense-d435.md) | The one hardware-measured component |

## Running the tests

    python3 -m pytest -q

**188 passed, 3 skipped** (Jetson Orin Nano, 2026-09-14).

On a machine with ROS 2 installed system-wide, plain pytest fails before
collecting anything:

    PluginValidationError: Plugin 'launch_testing' for hook
    'pytest_pycollect_makemodule' - Argument(s) {'path'} are declared in
    the hookimpl but can not be found in the hookspec

An environment collision, not a broken suite: ROS 2's launch_testing registers
a hook using a `path` argument that pytest 8 removed. `pytest.ini` disables the
pair so the suite runs with no arguments. Disabling only one makes it worse -
they reference each other's hooks. The entry-point name is `launch_ros`, which
does NOT match its module name `launch_testing_ros_pytest_entrypoint`. Blunt
fallback that always works:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q

## Status vocabulary

| Marker | Meaning |
|---|---|
| **CURRENT** | Implemented and exercised by tests |
| **PARTIALLY IMPLEMENTED** | Code exists; incomplete or not exercised on hardware |
| **VALIDATED** | Measured against real hardware, evidence committed |
| **PLANNED** | Designed; no implementation |
| **DEFERRED** | Known issue, intentionally not being worked on now |

## Code

| Path | Contents | Status |
|---|---|---|
| `common/` | Shared types (DepthKind, DetectedObject3D, LiDAR types), config, logging | CURRENT |
| `hardware/` | camera.py, realsense_d435.py, lidar/ (X4 Pro driver, scan processing, sectors) | PARTIALLY IMPLEMENTED |
| `runtime/` | Process manager, camera / perception / lidar processes, bounded IPC | CURRENT |
| `perception/` | detector/, depth/, tracking/, fusion.py | PARTIALLY IMPLEMENTED |
| `geometry/` | transforms.py - pixel to camera to world | CURRENT |
| `mapping/` | semantic_map.py | PARTIALLY IMPLEMENTED |
| `localization/` | base_pose_estimator.py | PLANNED - interface only |
| `navigation/` | base_planner.py | PLANNED - interface only |
| `swarm/` | communication_client.py | PLANNED - refused at runtime |
| `shaper/` | base_shaper_client.py | PLANNED - never a real-time dependency |
| `visualization/` | Dashboard, LiDAR view, perception view | PARTIALLY IMPLEMENTED |
| `tools/` | jetson_preflight.py, hardware_verification.py, validate_realsense_d435.py | CURRENT |
| `config/` | YAML: system, camera, perception, lidar, navigation, mqtt, robots, leader, safety | CURRENT |
| `validation/` | **Real hardware measurements** - D435 identity and factory intrinsics | **VALIDATED** |
| `tests/` | 188 passing, 3 skipped | CURRENT |

Entry points: main.py and run_leader.py.

## The D435 is validated but NOT in the pipeline

The most important thing to understand about the current state, and the
easiest to get wrong.

- The D435 is fitted; identity, firmware, USB mode and factory intrinsics are
  **measured and committed** (validation/).
- It produces **genuinely metric** depth.
- **Nothing in runtime/ or perception/ references it.** config/camera.yaml
  keeps it deliberately separate from the generic OpenCV source.

So perception runs on **RELATIVE** monocular depth while metric depth sits
available and unused beside it. An honest intermediate state - and exactly what
DepthKind exists to express: every depth frame carries its own unit, so the two
sources coexist without either being mistaken for the other.

Wiring the D435 into perception_process is the highest-value unstarted work.
The mechanism already exists: DepthKind propagates to
DetectedObject3D.coords_kind, so the pipeline would start producing metric
objects without a single consumer needing to be told.

## Invariants - do not break these

Enforced in code, not merely documented.

1. **Never fake sensor data.** No synthetic scans presented as real, no forced
   health flags.
2. **No silent hardware fallback.** system.mode live raises
   HardwareUnavailableError rather than returning a SyntheticCamera; MockLidar
   only by explicit simulation_mode true.
3. **Mock and relative depth are never metric.** DepthKind defaults to UNKNOWN.
4. **Bounded queues only.** LatestValueQueue is drop-oldest; reads are
   destructive, so one queue per producer/consumer pair.
5. **Safety is independent of AI.** ShapeR, reconstruction, visualization and
   database work are never on the real-time path.
6. **Swarm is refused, not merely disabled.**
7. **Motors stay disabled.** No motor-control code exists anywhere.
8. **Tests are never weakened to obtain a pass.**
9. **Nothing is reported as verified without hardware evidence.**

## Known gaps

| Gap | Note |
|---|---|
| D435 metric depth not consumed by the pipeline | highest-value unstarted work |
| Pipeline depth is relative, so 3D positions are not metric | see the depth ADR |
| LiDAR ~4 Hz measured; full-runtime scan reception | **DEFERRED** |
| No pose estimate, no SLAM, no planner | interfaces only |
| No end-to-end run on the robot | tests passing on the Jetson is not the same thing |
| No literature review conducted | no citations exist in this repository |
