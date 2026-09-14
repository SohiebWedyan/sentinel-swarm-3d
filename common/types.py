"""Core data types shared across the perception, geometry, mapping and
navigation layers.

Keeping these as plain dataclasses (rather than, say, dicts passed around
ad-hoc) is what lets every layer stay loosely coupled: a detector, a depth
estimator, a tracker or a mapper can all be swapped independently as long as
they still produce/consume these shapes.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


def new_object_id() -> str:
    """Generate a short, unique object id (used for detections/tracks)."""
    return uuid.uuid4().hex[:12]


def now() -> float:
    """Monotonic-ish wall clock timestamp (seconds, float)."""
    return time.time()


@dataclass(frozen=True)
class BBox:
    """Axis-aligned pixel bounding box, (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class TrackingState(str, Enum):
    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class Detection2D:
    """A single raw 2D detector output for one frame, pre-fusion.

    Geometry lives in `bbox` (one source of truth); the center/size
    accessors below are conveniences that delegate to it, so a detection
    can be consumed as x1/y1/x2/y2 or as center+size without two copies of
    the same numbers being able to disagree.
    """

    bbox: BBox
    class_id: int
    class_name: str
    confidence: float
    frame_timestamp: float = field(default_factory=now)

    @property
    def x1(self) -> float:
        return self.bbox.x1

    @property
    def y1(self) -> float:
        return self.bbox.y1

    @property
    def x2(self) -> float:
        return self.bbox.x2

    @property
    def y2(self) -> float:
        return self.bbox.y2

    @property
    def center_x(self) -> float:
        return self.bbox.center[0]

    @property
    def center_y(self) -> float:
        return self.bbox.center[1]

    @property
    def width(self) -> float:
        return self.bbox.width

    @property
    def height(self) -> float:
        return self.bbox.height

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
            "center_x": self.center_x, "center_y": self.center_y,
            "width": self.width, "height": self.height,
            "frame_timestamp": self.frame_timestamp,
        }


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    width: int = 0
    height: int = 0


@dataclass
class Pose3D:
    """A 6-DoF pose: position (meters) + orientation as roll/pitch/yaw (rad).

    Kept simple (RPY, not quaternion) for Phase 1 readability. The
    localization layer's PoseEstimator interface is free to carry a
    quaternion/covariance internally and only expose Pose3D at the boundary.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    frame: str = "world"
    timestamp: float = field(default_factory=now)


@dataclass
class DetectedObject3D:
    """The canonical fused perception output — one entry per tracked object.

    This is the schema referenced throughout the project spec: every
    detected object must carry an id, class, confidence, camera- and
    world-frame position, estimated size, timestamp, source robot and
    tracking state.
    """

    object_id: str
    class_id: int
    class_name: str
    confidence: float

    camera_coords: tuple[float, float, float]   # (X, Y, Z) meters, camera frame
    world_coords: Optional[tuple[float, float, float]] = None  # (X, Y, Z), world frame

    dimensions: Optional[tuple[float, float, float]] = None  # (w, h, d) meters, estimated
    velocity: Optional[tuple[float, float, float]] = None    # meters/sec, world frame

    bbox: Optional[BBox] = None
    depth_confidence: float = 0.0

    timestamp: float = field(default_factory=now)
    source_robot: str = "unknown"
    tracking_state: TrackingState = TrackingState.NEW

    track_id: Optional[int] = None
    observations: int = 1

    def to_dict(self) -> dict:
        """JSON-serializable representation (used by MQTT + SemanticMap)."""
        return {
            "object_id": self.object_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "camera_coords": list(self.camera_coords),
            "world_coords": list(self.world_coords) if self.world_coords else None,
            "dimensions": list(self.dimensions) if self.dimensions else None,
            "velocity": list(self.velocity) if self.velocity else None,
            "bbox": list(self.bbox.as_xyxy()) if self.bbox else None,
            "depth_confidence": self.depth_confidence,
            "timestamp": self.timestamp,
            "source_robot": self.source_robot,
            "tracking_state": self.tracking_state.value,
            "track_id": self.track_id,
            "observations": self.observations,
        }


# ---------------------------------------------------------------------------
# LiDAR types (leader-only runtime; YDLiDAR X4 Pro hardware).
#
# These follow the same "plain dataclass as the contract between layers"
# pattern as everything above: hardware/lidar/ driver implementations
# produce LidarScanPacket, hardware/lidar/scan_processor.py filters it
# (still a LidarScanPacket), and hardware/lidar/obstacle_sectors.py reduces
# it to SectorClearance per LidarSectorName — no layer invents its own
# ad-hoc shape, exactly like DetectedObject3D above.
#
# UNITS AND ANGLE CONVENTION (one internal representation, documented once):
#   * distances are ALWAYS meters
#   * the stored angle is ALWAYS signed degrees in [-180, 180), with
#     0 = straight ahead (robot forward) and positive = counter-clockwise
#     (to the robot's left)
#   * LidarPoint.angle_rad is the radian view of that same angle, provided
#     because navigation/geometry math wants radians while the visualization
#     and sector config are far more readable in degrees. Degrees is the
#     stored form; radians is derived. Drivers convert into this convention
#     (the YDLiDAR SDK reports radians) before constructing a LidarPoint,
#     so nothing downstream needs to know what the sensor natively emits.
# ---------------------------------------------------------------------------


class LidarSectorName(str, Enum):
    FRONT = "front"
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    LEFT = "left"
    RIGHT = "right"
    REAR = "rear"


class LidarSafetyLevel(str, Enum):
    """Safety banding derived from LiDAR clearance alone.

    Published for consumers to act on later — nothing in the current
    runtime actuates on it, because no motor-control code exists yet
    (Phase 1E/1F). NO_DATA is distinct from NORMAL on purpose: "the LiDAR
    told us the path is clear" and "the LiDAR told us nothing" must never
    collapse into the same value.
    """

    NO_DATA = "no_data"
    NORMAL = "normal"
    CAUTION = "caution"
    STOP = "stop"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class LidarPoint:
    """One ranging measurement. See the angle convention note above."""

    angle_deg: float
    distance_m: float
    intensity: Optional[float] = None  # raw sensor intensity when the driver reports one
    timestamp: Optional[float] = None  # per-point time when available, else the packet's

    @property
    def angle_rad(self) -> float:
        return math.radians(self.angle_deg)


@dataclass
class LidarScanPacket:
    """One full (or partial) rotation's worth of LidarPoints, plus the
    acquisition metadata the health/diagnostic paths need.

    valid_points/invalid_points are filled in by ScanProcessor: `points`
    only ever holds the valid ones, and invalid_points records how many
    were dropped, so a scan that is mostly garbage is visible as such
    instead of silently looking like a sparse room.
    """

    points: list[LidarPoint]
    timestamp: float = field(default_factory=now)
    sequence_id: int = 0
    frame_id: str = "lidar"
    source: str = "lidar"
    scan_duration_s: float = 0.0
    scan_frequency_hz: float = 0.0
    valid_points: int = 0
    invalid_points: int = 0


@dataclass
class SectorClearance:
    """Obstacle summary for one angular sector of a single scan."""

    sector: LidarSectorName
    min_distance_m: Optional[float]
    median_distance_m: Optional[float]
    num_valid_points: int
    obstacle_detected: bool


@dataclass
class LidarStatus:
    """Everything the diagnostic/monitoring paths need to answer "is the
    LiDAR actually working, and if not, why".

    Deliberately a value object with no live references into a driver, so
    it can be published across a process boundary
    (runtime.shared_state.LatestValueQueue) and rendered by the
    visualization layer without either side touching the SDK.
    """

    connected: bool = False
    scanning: bool = False
    healthy: bool = False
    port: Optional[str] = None
    baudrate: int = 0
    model: str = ""
    last_scan_timestamp: Optional[float] = None
    #: MEASURED rotation rate, derived from actual scan arrival times.
    scan_rate_hz: float = 0.0
    #: CONFIGURED expectation (config/lidar.yaml expected_scan_frequency_hz).
    #: Reported alongside the measured rate so diagnostics can show the two
    #: side by side — the device's real rate is set by its motor, not by
    #: anything this project commands, so they are allowed to differ.
    expected_scan_rate_hz: float = 0.0
    error_message: Optional[str] = None
    simulated: bool = False

    def to_dict(self) -> dict:
        return {
            "connected": self.connected,
            "scanning": self.scanning,
            "healthy": self.healthy,
            "port": self.port,
            "baudrate": self.baudrate,
            "model": self.model,
            "last_scan_timestamp": self.last_scan_timestamp,
            "scan_rate_hz": self.scan_rate_hz,
            "expected_scan_rate_hz": self.expected_scan_rate_hz,
            "error_message": self.error_message,
            "simulated": self.simulated,
        }


# ---------------------------------------------------------------------------
# Camera + perception pipeline types (Phase 2).
#
# Same contract-between-layers pattern as the LiDAR types above:
# runtime/camera_process.py produces FramePacket, runtime/perception_process.py
# consumes it and produces PerceptionResult, and the visualization layer
# consumes that. Nothing carries a live handle across a process boundary —
# these are plain, picklable values.
# ---------------------------------------------------------------------------


class DepthKind(str, Enum):
    """What a depth map's numbers actually MEAN.

    This distinction is load-bearing, not bookkeeping. Monocular models like
    Depth Anything produce RELATIVE depth: internally consistent within one
    frame, but unitless and up to an unknown (per-frame) scale. Treating
    those numbers as meters silently produces confident, wrong 3D positions.

    Anything that wants metric distance must either use a METRIC source
    (stereo, RGB-D, a metric-finetuned model) or first solve for scale using
    a known reference — and until it does, it should refuse to publish
    metric claims. `UNKNOWN` exists so an unlabelled source can never be
    mistaken for a metric one by default.
    """

    METRIC = "metric"        # values are meters
    RELATIVE = "relative"    # unitless, monotonic with distance, unknown scale
    UNKNOWN = "unknown"


@dataclass
class FramePacket:
    """One camera frame crossing a process boundary, with the metadata the
    consumer needs to reason about freshness and geometry.

    `scale_from_capture` is a trap-avoidance field: if a frame was
    downscaled before IPC (config camera.ipc.scale), the camera intrinsics
    in config/camera.yaml no longer describe it. Any consumer doing
    projection math must scale fx/fy/cx/cy by this factor. 1.0 means the
    frame is at native capture resolution and intrinsics apply as-is.
    """

    image: "np.ndarray"  # type: ignore[name-defined]  # HxWx3 BGR uint8
    timestamp: float = field(default_factory=now)
    sequence_id: int = 0
    frame_id: str = "camera"
    source: str = "camera"
    width: int = 0
    height: int = 0
    capture_fps: float = 0.0
    scale_from_capture: float = 1.0

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.timestamp)


@dataclass
class CameraStatus:
    """Camera health/diagnostics snapshot (mirrors LidarStatus's role)."""

    connected: bool = False
    streaming: bool = False
    healthy: bool = False
    device: str = ""
    width: int = 0
    height: int = 0
    configured_fps: float = 0.0
    measured_fps: float = 0.0
    last_frame_timestamp: Optional[float] = None
    sequence_id: int = 0
    frames_dropped: int = 0
    reconnect_attempts: int = 0
    error_message: Optional[str] = None
    simulated: bool = False

    def to_dict(self) -> dict:
        return {
            "connected": self.connected,
            "streaming": self.streaming,
            "healthy": self.healthy,
            "device": self.device,
            "width": self.width,
            "height": self.height,
            "configured_fps": self.configured_fps,
            "measured_fps": self.measured_fps,
            "last_frame_timestamp": self.last_frame_timestamp,
            "sequence_id": self.sequence_id,
            "frames_dropped": self.frames_dropped,
            "reconnect_attempts": self.reconnect_attempts,
            "error_message": self.error_message,
            "simulated": self.simulated,
        }


@dataclass
class DepthFrame:
    """A depth map plus the provenance needed to interpret it.

    `kind` is mandatory and has no default — a depth map that doesn't say
    what its units are is exactly the thing this type exists to prevent.
    """

    depth: "np.ndarray"  # type: ignore[name-defined]  # HxW float32
    kind: DepthKind
    model: str = ""
    device: str = ""
    inference_ms: float = 0.0
    simulated: bool = False
    confidence: Optional["np.ndarray"] = None  # type: ignore[name-defined]

    @property
    def height(self) -> int:
        return int(self.depth.shape[0]) if self.depth is not None else 0

    @property
    def width(self) -> int:
        return int(self.depth.shape[1]) if self.depth is not None else 0

    @property
    def is_metric(self) -> bool:
        """Only ever True for a source that actually reports meters."""
        return self.kind is DepthKind.METRIC

    def describe_units(self) -> str:
        if self.kind is DepthKind.METRIC:
            return "meters"
        if self.kind is DepthKind.RELATIVE:
            return "relative (unitless, unknown scale)"
        return "unknown units"


@dataclass
class PerceptionTiming:
    """Per-stage timing for one processed frame, in milliseconds."""

    detector_ms: float = 0.0
    depth_ms: float = 0.0
    total_ms: float = 0.0
    #: capture timestamp -> result ready, i.e. what a consumer actually waited.
    pipeline_latency_ms: float = 0.0


@dataclass
class PerceptionResult:
    """The synchronized output of one frame through YOLO + depth.

    Both branches ran on the SAME frame (same sequence_id/timestamp) — that
    synchronization is the point of the perception process, and is why this
    is one object rather than two independently-published streams.
    """

    frame_id: str = "camera"
    timestamp: float = field(default_factory=now)
    sequence_id: int = 0
    detections: list = field(default_factory=list)  # list[Detection2D]
    depth: Optional[DepthFrame] = None
    image_width: int = 0
    image_height: int = 0
    timing: PerceptionTiming = field(default_factory=PerceptionTiming)
    #: Optional copy of the frame, published only for visualization
    #: (config perception.process.publish_frame). None keeps IPC cheap.
    image: Optional["np.ndarray"] = None  # type: ignore[name-defined]

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    def to_dict(self) -> dict:
        """Metadata-only view (no pixel data) — safe to log or publish."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "sequence_id": self.sequence_id,
            "detection_count": self.detection_count,
            "detections": [d.to_dict() for d in self.detections],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "has_depth": self.has_depth,
            "depth_kind": self.depth.kind.value if self.depth else None,
            "detector_ms": self.timing.detector_ms,
            "depth_ms": self.timing.depth_ms,
            "total_ms": self.timing.total_ms,
            "pipeline_latency_ms": self.timing.pipeline_latency_ms,
        }


@dataclass
class PerceptionHealth:
    """Perception process health/diagnostics snapshot."""

    healthy: bool = False
    detector_ready: bool = False
    detector_degraded: bool = False
    depth_ready: bool = False
    depth_simulated: bool = False
    depth_kind: DepthKind = DepthKind.UNKNOWN
    fps: float = 0.0
    last_result_timestamp: Optional[float] = None
    frames_processed: int = 0
    frames_dropped_stale: int = 0
    detector_model: str = ""
    depth_model: str = ""
    device: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "detector_ready": self.detector_ready,
            "detector_degraded": self.detector_degraded,
            "depth_ready": self.depth_ready,
            "depth_simulated": self.depth_simulated,
            "depth_kind": self.depth_kind.value,
            "fps": self.fps,
            "last_result_timestamp": self.last_result_timestamp,
            "frames_processed": self.frames_processed,
            "frames_dropped_stale": self.frames_dropped_stale,
            "detector_model": self.detector_model,
            "depth_model": self.depth_model,
            "device": self.device,
            "error_message": self.error_message,
        }
