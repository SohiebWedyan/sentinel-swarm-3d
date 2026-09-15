# Intel RealSense D435

**Status: VALIDATED on real hardware - but NOT consumed by the perception
pipeline.**

This is the only component in the repository with committed hardware
measurements. It is also not wired into anything yet. Both halves matter.

## Measured, not assumed

From validation/d435_validation_20260913.json, produced by
tools/validate_realsense_d435.py against the physical unit:

| Property | Measured value |
|---|---|
| Device | RealSense **D435** - explicitly checked as *not* D435i |
| Serial | 215322073372 |
| Firmware | 5.12.7.150 |
| USB mode | **3.2** |

### Factory intrinsics at 640x480

| | fx | fy | ppx | ppy |
|---|---|---|---|---|
| **Colour** | 605.0287 | 604.6178 | 322.2229 | 248.9775 |
| **Depth** | 386.1925 | 386.1925 | 326.1069 | 238.5620 |

Also on file: camera_calib_d435.npz (2.5 KB - intrinsics only, no captured
frames).

## Do not calibrate this camera

The D435 is **factory calibrated**, and its distortion coefficients are all
zero because correction happens in firmware.

**Never apply cv2.undistort to D435 output.** Doing so applies a second
correction to an already-corrected image and quietly degrades every projection
that follows.

This is a real difference from the generic USB camera path, whose intrinsics in
config/camera.yaml are an *estimate* and do need calibration before its
projection output can be trusted.

## No IMU

This is a **D435**, not a D435i. It has no internal IMU. Any design that
assumes camera-provided inertial data does not apply here - the IMU remains a
separate, planned component.

The validation tool checks this explicitly, because the two models are visually
near-identical and the difference is silent until something downstream waits
forever for IMU frames that never arrive.

## USB sensitivity

Depth has been observed to fail to start (Depth stream start failure) until
other USB devices were unplugged. Give the D435 a dedicated port; consider a
powered hub.

The symptom is misleading: colour streams fine while depth silently does not
start, which reads as a driver or configuration fault rather than a bus one.

## Deliberately outside the perception pipeline

config/camera.yaml keeps the D435 block separate from the generic OpenCV
source, and **nothing in runtime/ or perception/ references it**:

    grep -rn "realsense\|D435" runtime/ perception/   ->   no matches

So the pipeline runs on the generic camera and RELATIVE monocular depth, while
this device's METRIC depth sits available and unused.

That is a deliberate intermediate state, not an oversight. Stated precisely:

- The physical D435 unit is **VALIDATED** against real hardware.
- The D435 **produces METRIC depth**.
- The runtime and perception pipeline **do NOT consume** that metric depth.
- The pipeline operates on **RELATIVE** monocular depth.
- Therefore the current perception pipeline must **not** be characterised as
  metric, and no measurement derived from it may be reported in metres.

DepthKind carries this distinction in code - METRIC, RELATIVE, UNKNOWN - with
UNKNOWN as the default, so a source that does not declare its unit is never
assumed to be metric.

## What wiring it in would unblock

Metric depth removes the single largest limitation in the project: with
DepthKind.METRIC flowing into Object3DFusion, fused object coordinates become
real metres, and everything downstream that was blocked on scale - semantic map
merging by position, obstacle sizing, formation geometry - becomes meaningful.

The mechanism already exists. DepthKind is carried on every depth frame and
propagated to DetectedObject3D.coords_kind, so the pipeline would start
producing metric objects without a single consumer needing to be told.

## Verification

    python3 tools/validate_realsense_d435.py

Re-running writes a fresh report. Keep validation output under validation/,
**not** under logs/ - .gitignore excludes logs/, and evidence that gets ignored
is evidence that disappears.
