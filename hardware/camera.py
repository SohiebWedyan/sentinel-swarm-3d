"""Camera hardware abstraction.

Three concrete sources, selected purely by config (config/camera.yaml ->
source_type), so nothing above this module needs to know or care whether
frames come from a real device, a video file, or a synthetic generator:

- DeviceCamera:     a real OpenCV-indexed camera (USB webcam / CSI via v4l2)
- VideoFileCamera:  loops a recorded .mp4/.avi — useful for repeatable tests
- SyntheticCamera:  no OpenCV/hardware required at all; generates moving
                     shapes on a blank frame so the whole pipeline (including
                     depth + fusion + visualization) is runnable on a laptop
                     with zero physical hardware and zero video assets.

All three implement BaseCamera, which is what perception/ and main.py
actually depend on.
"""

from __future__ import annotations

import logging
import sys
import time
from abc import abstractmethod
from typing import Optional

import numpy as np

from common.types import CameraIntrinsics
from hardware.base import BaseSensor, HardwareUnavailableError

logger = logging.getLogger(__name__)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover - exercised only when opencv is absent
    _HAS_CV2 = False
    logger.warning("opencv-python not available; only SyntheticCamera will work.")


class Frame:
    """A single captured frame plus its capture timestamp."""

    __slots__ = ("image", "timestamp", "frame_id")

    def __init__(self, image: np.ndarray, timestamp: float, frame_id: int) -> None:
        self.image = image
        self.timestamp = timestamp
        self.frame_id = frame_id


class BaseCamera(BaseSensor):
    """Camera-specific extension of BaseSensor: read() returns a Frame and
    every camera exposes its intrinsics so the geometry layer never has to
    guess where they came from.
    """

    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        self.intrinsics = intrinsics
        self._frame_counter = 0

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next Frame, or None if the stream ended (video file) /
        a transient read failure occurred (caller should keep polling).
        """

    def describe(self) -> str:
        """Human-readable identification of the actual frame source in use.

        The factory can fall back (a requested webcam that isn't there
        becomes synthetic frames), so the operator must be able to see what
        is REALLY feeding the pipeline rather than what was requested.
        """
        return self.__class__.__name__

    @property
    def is_simulated(self) -> bool:
        """True when frames are generated rather than captured."""
        return False

    def _next_frame_id(self) -> int:
        self._frame_counter += 1
        return self._frame_counter


class DeviceCamera(BaseCamera):
    """Real camera via OpenCV VideoCapture (USB/CSI webcam).

    `device` is either an integer OpenCV index (0, 1, ...) or an explicit
    device path ("/dev/video0"). OpenCV accepts both; the path form is
    preferred on a machine with several cameras because indices can be
    reassigned across reboots.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        device_index: int,
        width: int,
        height: int,
        fps: int,
        device_path: str | None = None,
        backend: str = "auto",
        fourcc: str = "MJPG",
        buffer_size: int = 1,
    ):
        super().__init__(intrinsics)
        self.device_index = device_index
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.fourcc = fourcc
        self.buffer_size = buffer_size
        self._cap = None
        self.negotiated_fourcc: str = ""
        self.negotiated_fps: float = 0.0

    @property
    def device(self):
        """Whatever should actually be handed to cv2.VideoCapture."""
        return self.device_path if self.device_path else self.device_index

    def describe(self) -> str:
        return f"DeviceCamera({self.device})"

    def _resolve_backend(self) -> int:
        """V4L2 explicitly on Linux rather than letting OpenCV choose.

        Left to itself OpenCV may pick the FFMPEG backend for a camera
        device, which ignores most CAP_PROP settings (format, buffer size)
        and warns that it needs libavdevice. V4L2 is the backend that
        actually honors the tuning below.
        """
        if not _HAS_CV2:
            return 0
        backends = {
            "v4l2": getattr(cv2, "CAP_V4L2", 0),
            "any": getattr(cv2, "CAP_ANY", 0),
            "gstreamer": getattr(cv2, "CAP_GSTREAMER", 0),
        }
        if self.backend != "auto":
            return backends.get(self.backend.lower(), getattr(cv2, "CAP_ANY", 0))
        if sys.platform.startswith("linux"):
            return backends["v4l2"]
        return backends["any"]

    @staticmethod
    def _decode_fourcc(value: float) -> str:
        code = int(value)
        if code <= 0:
            return ""
        return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))

    def open(self) -> None:
        if not _HAS_CV2:
            raise HardwareUnavailableError("opencv-python is not installed")

        # Idempotent: re-opening (e.g. a reconnect, or a factory that already
        # probed the device) must not leak the previous capture handle.
        self.close()

        api_preference = self._resolve_backend()
        self._cap = cv2.VideoCapture(self.device, api_preference) if api_preference \
            else cv2.VideoCapture(self.device)

        if not self._cap.isOpened():
            raise HardwareUnavailableError(f"Could not open camera device {self.device}")

        # ORDER MATTERS: pixel format must be set BEFORE resolution, or many
        # V4L2 drivers ignore it and quietly keep the previous format.
        #
        # Why this is not cosmetic: a USB webcam defaults to uncompressed
        # YUYV, whose bandwidth caps 1280x720 at roughly 5-10 FPS over USB
        # 2.0. MJPG is compressed and reaches the full 30 FPS at the same
        # resolution. Getting this wrong looks like "the camera is slow"
        # rather than "the pixel format is wrong".
        if self.fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # V4L2 keeps a queue of captured frames and read() returns the
        # OLDEST one. With the default depth (often 4) every frame handed to
        # the pipeline is already ~4 frame-times stale — about 130 ms at
        # 30 FPS — which silently defeats the whole latest-frame-only design
        # upstream. A depth of 1 keeps read() close to "now".
        if self.buffer_size:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.negotiated_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.negotiated_fourcc = self._decode_fourcc(self._cap.get(cv2.CAP_PROP_FOURCC) or 0)

        if (self.width and actual_w and actual_w != self.width) or \
           (self.height and actual_h and actual_h != self.height):
            # A webcam that doesn't support the requested mode silently gives
            # you a different one — say so rather than letting the operator
            # assume the configured resolution is what's flowing.
            logger.warning("Camera %s did not accept %dx%d; actually delivering %dx%d.",
                           self.device, self.width, self.height, actual_w, actual_h)
        if actual_w:
            self.width = actual_w
        if actual_h:
            self.height = actual_h

        if self.fourcc and self.negotiated_fourcc and \
                self.negotiated_fourcc.upper() != self.fourcc.upper():
            logger.warning(
                "Camera %s did not accept pixel format %s; negotiated %s instead. "
                "Uncompressed formats (YUYV) severely limit FPS at high resolution — "
                "check `v4l2-ctl --list-formats-ext -d %s` for what this device supports.",
                self.device, self.fourcc, self.negotiated_fourcc, self.device,
            )

        if self.fps and self.negotiated_fps and self.negotiated_fps < self.fps * 0.9:
            logger.warning("Camera %s negotiated %.1f FPS, below the requested %d FPS.",
                           self.device, self.negotiated_fps, self.fps)

        logger.info("DeviceCamera opened on %s (%dx%d, fourcc=%s, %.1f FPS, buffer=%d)",
                    self.device, self.width, self.height,
                    self.negotiated_fourcc or "unknown", self.negotiated_fps,
                    self.buffer_size)

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            raise HardwareUnavailableError("DeviceCamera.read() called before open()")
        ok, image = self._cap.read()
        if not ok:
            logger.warning("DeviceCamera: frame grab failed")
            return None
        return Frame(image=image, timestamp=time.time(), frame_id=self._next_frame_id())

    def is_healthy(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoFileCamera(BaseCamera):
    """Plays back a recorded video file, looping when it reaches the end."""

    def __init__(self, intrinsics: CameraIntrinsics, path: str, loop: bool = True):
        super().__init__(intrinsics)
        self.path = path
        self.loop = loop
        self._cap = None

    def describe(self) -> str:
        return f"VideoFileCamera({self.path})"

    @property
    def is_simulated(self) -> bool:
        return True

    def open(self) -> None:
        if not _HAS_CV2:
            raise HardwareUnavailableError("opencv-python is not installed")
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise HardwareUnavailableError(f"Could not open video file: {self.path}")
        logger.info("VideoFileCamera opened %s", self.path)

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            raise HardwareUnavailableError("VideoFileCamera.read() called before open()")
        ok, image = self._cap.read()
        if not ok:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, image = self._cap.read()
                if not ok:
                    return None
            else:
                return None
        return Frame(image=image, timestamp=time.time(), frame_id=self._next_frame_id())

    def is_healthy(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SyntheticCamera(BaseCamera):
    """Zero-dependency frame source for simulation mode / CI / demos without
    any camera, video file, or even OpenCV installed.

    Draws a few moving rectangles (stand-ins for "objects") on a blank
    canvas using pure numpy, so the perception pipeline has something
    non-trivial to detect a bounding box around even when Ultralytics/OpenCV
    fall back to their own mock behavior.
    """

    def __init__(self, intrinsics: CameraIntrinsics, width: int = 1280, height: int = 720, num_shapes: int = 3):
        super().__init__(intrinsics)
        self.width = width
        self.height = height
        self.num_shapes = num_shapes
        self._t0 = time.time()
        self._open = False

    def describe(self) -> str:
        return f"SyntheticCamera({self.width}x{self.height}, generated frames)"

    @property
    def is_simulated(self) -> bool:
        return True

    def open(self) -> None:
        self._open = True
        logger.info("SyntheticCamera started (%dx%d, %d synthetic objects)",
                    self.width, self.height, self.num_shapes)

    def read(self) -> Optional[Frame]:
        if not self._open:
            raise HardwareUnavailableError("SyntheticCamera.read() called before open()")

        t = time.time() - self._t0
        image = np.full((self.height, self.width, 3), 30, dtype=np.uint8)  # dark gray bg

        for i in range(self.num_shapes):
            phase = i * (2 * np.pi / max(self.num_shapes, 1))
            cx = int(self.width / 2 + (self.width / 3) * np.sin(t * 0.5 + phase))
            cy = int(self.height / 2 + (self.height / 4) * np.cos(t * 0.7 + phase))
            size = 40 + 20 * (i + 1)
            x1, y1 = max(cx - size, 0), max(cy - size, 0)
            x2, y2 = min(cx + size, self.width - 1), min(cy + size, self.height - 1)
            color = [(60, 180, 255), (80, 220, 100), (220, 90, 90)][i % 3]
            image[y1:y2, x1:x2] = color

        return Frame(image=image, timestamp=time.time(), frame_id=self._next_frame_id())

    def is_healthy(self) -> bool:
        return self._open

    def close(self) -> None:
        self._open = False


#: Device nodes scanned when no explicit device_path is configured. Only
#: even indices are real capture devices on many systems (odd ones are
#: metadata nodes), but we probe rather than assume.
def discover_video_devices() -> list[str]:
    """Enumerate /dev/video* that exist on this machine, lowest first.

    Returns [] on a machine with no cameras — which is information, not an
    error. Callers decide whether that is fatal.
    """
    import glob
    import re

    def _index(path: str) -> int:
        match = re.search(r"(\d+)$", path)
        return int(match.group(1)) if match else 9999

    return sorted(glob.glob("/dev/video*"), key=_index)


def probe_working_video_device(
    intrinsics: CameraIntrinsics,
    candidates: list[str],
    width: int,
    height: int,
    fps: int,
    backend: str = "auto",
    fourcc: str = "MJPG",
    buffer_size: int = 1,
) -> Optional["DeviceCamera"]:
    """Open each candidate and keep the first that actually DELIVERS A FRAME.

    Opening succeeds on plenty of nodes that never produce an image (a
    metadata node, a device held by another process), so "opened" is not
    evidence of a working camera — a decoded frame is. The returned camera
    is left open and ready.
    """
    for path in candidates:
        camera = DeviceCamera(intrinsics, 0, width, height, fps, device_path=path,
                              backend=backend, fourcc=fourcc, buffer_size=buffer_size)
        try:
            camera.open()
        except HardwareUnavailableError:
            continue

        frame = None
        for _ in range(5):          # first grab after open is often empty
            frame = camera.read()
            if frame is not None:
                break

        if frame is not None:
            logger.info("Discovered working camera at %s", path)
            return camera

        logger.debug("%s opened but delivered no frames; trying the next device.", path)
        camera.close()

    return None


def build_camera_from_config(cfg: dict) -> BaseCamera:
    """Factory: config/camera.yaml + system.mode -> a concrete BaseCamera.

    MODE IS AUTHORITATIVE, and the two modes differ in one critical way:

        system.mode: "live"        a real camera is REQUIRED. If none can be
                                   found, this raises HardwareUnavailableError.
                                   It NEVER returns a SyntheticCamera.

        system.mode: "simulation"  synthetic frames are acceptable, and a
                                   real webcam is used opportunistically if
                                   simulation.kind says so.

    The asymmetry is deliberate. A silent fallback from "real camera" to
    "generated frames" is how a system ends up reporting healthy perception
    while looking at nothing — and how a hardware-verification run could
    produce a false pass. In live mode the failure is loud.
    """
    cam_cfg = cfg["camera"]
    intr_cfg = cam_cfg["intrinsics"]
    intrinsics = CameraIntrinsics(
        fx=intr_cfg["fx"], fy=intr_cfg["fy"], cx=intr_cfg["cx"], cy=intr_cfg["cy"],
        distortion=tuple(intr_cfg.get("distortion", [0, 0, 0, 0, 0])),
        width=cam_cfg.get("width", 0), height=cam_cfg.get("height", 0),
    )

    mode = str(cfg.get("system", {}).get("mode", "simulation")).lower()
    source_type = cam_cfg.get("source_type", "simulation")
    device_path = cam_cfg.get("device_path") or None
    #: None (or absent/null in YAML) means "auto-discover"; an integer is an
    #: explicit operator choice and is honored as configured.
    device_index = cam_cfg.get("device_index")
    hw_cfg = cam_cfg.get("hardware", {})
    width = cam_cfg.get("width", 1280)
    height = cam_cfg.get("height", 720)
    fps = cam_cfg.get("fps", 30)
    backend = hw_cfg.get("backend", "auto")
    fourcc = hw_cfg.get("fourcc", "MJPG")
    buffer_size = hw_cfg.get("buffer_size", 1)

    def _device_camera() -> DeviceCamera:
        return DeviceCamera(
            intrinsics, device_index if device_index is not None else 0,
            width, height, fps,
            device_path=device_path, backend=backend, fourcc=fourcc,
            buffer_size=buffer_size,
        )

    # -- LIVE: a real camera or nothing --------------------------------
    if mode == "live" or source_type == "device":
        # An explicitly configured device is used as-is. Construction stays
        # pure: if that device is wrong or absent, open() fails loudly later
        # (camera_process surfaces it as an unhealthy camera with the reason).
        if device_path:
            logger.info("LIVE camera: using configured device_path %s", device_path)
            return _device_camera()

        if device_index is not None:
            logger.info("LIVE camera: using configured device_index %s", device_index)
            return _device_camera()

        # Nothing configured -> discover rather than guess /dev/video0.
        # This is the one path that probes, because picking a device without
        # checking it is exactly the guessing we want to avoid.
        candidates = discover_video_devices()
        if not candidates:
            raise HardwareUnavailableError(
                "LIVE mode requires a real camera but no /dev/video* devices exist. "
                "Plug in the USB camera (check `ls -l /dev/video*` and `dmesg | tail`), "
                "or set system.mode: \"simulation\" to run on synthetic frames."
            )

        logger.info("LIVE camera: probing discovered devices %s", ", ".join(candidates))
        camera = probe_working_video_device(
            intrinsics, candidates, width, height, fps, backend, fourcc, buffer_size
        )
        if camera is None:
            raise HardwareUnavailableError(
                f"LIVE mode requires a real camera. Found {', '.join(candidates)} but none "
                f"delivered a frame. Check permissions (`ls -l /dev/video*`, add your user "
                f"to the `video` group), that no other process holds the device, and "
                f"`v4l2-ctl --list-formats-ext -d <dev>` for supported formats."
            )
        return camera

    if source_type == "video_file":
        return VideoFileCamera(intrinsics, cam_cfg["video_file"])

    # -- SIMULATION: synthetic frames are acceptable --------------------
    # Falling back here is fine precisely because the mode says simulated
    # data is expected. The fallback is still logged, never silent.
    sim_cfg = cam_cfg.get("simulation", {})
    kind = sim_cfg.get("kind", "synthetic")

    if kind == "webcam" and _HAS_CV2:
        try:
            cam = _device_camera()
            cam.open()
            return cam
        except HardwareUnavailableError:
            logger.warning("Simulation kind 'webcam' requested but no webcam found; "
                           "falling back to synthetic frames. (This fallback exists ONLY "
                           "in simulation mode — system.mode: \"live\" would fail instead.)")

    if kind == "video_file" and _HAS_CV2:
        try:
            cam = VideoFileCamera(intrinsics, sim_cfg.get("fallback_video", ""))
            cam.open()
            return cam
        except HardwareUnavailableError:
            logger.warning("Simulation kind 'video_file' requested but file could not be "
                           "opened; falling back to synthetic frames.")

    return SyntheticCamera(intrinsics, cam_cfg.get("width", 1280), cam_cfg.get("height", 720))
