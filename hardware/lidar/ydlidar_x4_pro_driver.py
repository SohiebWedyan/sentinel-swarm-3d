"""YDLiDAR X4 Pro driver, built on the official YDLiDAR SDK.

    BaseLidar  ->  YDLidarX4ProDriver  ->  YDLidar SDK  ->  X4 Pro hardware

Every `ydlidar.*` call in this project lives in this file. Nothing else
imports the SDK, so the rest of the system talks only to BaseLidar and
common.types — swapping sensors later means writing a sibling module, not
touching callers.

The serial packet protocol is deliberately NOT reimplemented here: the SDK
owns framing, checksums, motor control, and device health. This module owns
configuration validation, port resolution, the X4 Pro's fixed hardware
profile, unit/convention normalization, and the failure policy.

FAILURE POLICY (important): if the SDK is missing, the port can't be found,
initialization fails, or scans stop arriving, this driver reports itself
unhealthy with a specific error_message and returns no scans. It never
substitutes synthetic data for real data — a robot acting on invented
obstacle readings it believes are real is far more dangerous than a robot
that knows its LiDAR is down. Simulation is an explicit, separate choice
(MockLidar, via simulation_mode: true).

SDK INSTALL (not a PyPI package):
    git clone https://github.com/YDLIDAR/YDLidar-SDK
    cd YDLidar-SDK && mkdir build && cd build && cmake .. && make && sudo make install
    cd ../ && pip install .
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from common.types import LidarPoint, LidarScanPacket, LidarStatus
from hardware.lidar.base_lidar import BaseLidar

logger = logging.getLogger(__name__)

try:
    import ydlidar  # official YDLidar-SDK python binding

    _HAS_YDLIDAR_SDK = True
except ImportError:  # pragma: no cover - exercised on any machine without the SDK
    ydlidar = None  # type: ignore[assignment]
    _HAS_YDLIDAR_SDK = False


class LidarConfigurationError(ValueError):
    """config/lidar.yaml describes something the X4 Pro cannot actually do.

    Raised at construction time rather than logged-and-ignored: a wrong
    baudrate or a ToF/single-channel misconfiguration doesn't degrade
    gracefully on this device, it produces silence or garbage. Better to
    fail loudly at startup than to debug an empty scan later.
    """


@dataclass(frozen=True)
class X4ProProfile:
    """Fixed hardware facts about the YDLiDAR X4 Pro.

    These are properties of the device, not user preferences, so they are
    validated against rather than read from config — config/lidar.yaml
    still spells them out for readability, but a value that contradicts
    this profile is rejected.

    `single_channel` is the one field here that cannot be confirmed from
    documentation alone and must be validated against the physical unit:
    it controls whether the SDK expects the device-info/health handshake on
    connect, and getting it wrong typically shows up as a failed
    initialize() or a connected device that streams nothing. If bench
    testing contradicts this profile, change it HERE (one line) — the
    validation, the SDK option, and config/lidar.yaml all derive from it.

    scan_frequency_min_hz/max_hz describe the rotation band the hardware
    can operate in. They bound the *expected* frequency the operator
    configures; see expected_scan_frequency_hz on the driver for why that
    is an expectation rather than a command.
    """

    model: str = "X4_PRO"
    baudrate: int = 128000
    sample_rate_khz: int = 5
    range_min_m: float = 0.12
    range_max_m: float = 10.0
    scan_angle_deg: float = 360.0
    scan_frequency_min_hz: float = 6.0
    scan_frequency_max_hz: float = 12.0
    lidar_type: str = "triangle"
    single_channel: bool = True
    is_tof: bool = False


X4_PRO = X4ProProfile()

#: Checked in order, after whatever the SDK's own port enumeration reports.
DEFAULT_PORT_CANDIDATES: tuple[str, ...] = (
    "/dev/ydlidar",   # the udev symlink YDLIDAR's own rules install — preferred
    "/dev/ttyUSB0",
    "/dev/ttyUSB1",
    "/dev/ttyACM0",
    "/dev/ttyACM1",
)


def validate_x4_pro_config(
    baudrate: int,
    lidar_type: str,
    single_channel: bool,
    is_tof: bool,
    min_distance_m: float,
    max_distance_m: float,
    sample_rate_khz: int,
    expected_scan_frequency_hz: float,
    profile: X4ProProfile = X4_PRO,
) -> None:
    """Reject configurations the X4 Pro cannot honor. Raises
    LidarConfigurationError with a message naming the offending field.

    Messages state what the profile requires rather than describing the
    device's internals, so a future profile correction doesn't leave
    stale claims about the hardware embedded in error strings.
    """
    if baudrate != profile.baudrate:
        raise LidarConfigurationError(
            f"YDLiDAR {profile.model} requires baudrate {profile.baudrate}, got {baudrate}. "
            f"The X4 Pro will not communicate at any other rate."
        )

    if str(lidar_type).lower() != profile.lidar_type:
        raise LidarConfigurationError(
            f"YDLiDAR {profile.model} is a '{profile.lidar_type}' (TYPE_TRIANGLE) LiDAR, "
            f"got lidar_type='{lidar_type}'."
        )

    if bool(single_channel) != profile.single_channel:
        raise LidarConfigurationError(
            f"YDLiDAR {profile.model} requires single_channel={profile.single_channel}, "
            f"got {single_channel}."
        )

    if bool(is_tof) != profile.is_tof:
        raise LidarConfigurationError(
            f"YDLiDAR {profile.model} is a triangulation device, not ToF: is_tof must be "
            f"{profile.is_tof}, got {is_tof}."
        )

    if sample_rate_khz != profile.sample_rate_khz:
        raise LidarConfigurationError(
            f"YDLiDAR {profile.model} runs at {profile.sample_rate_khz} kHz sample rate, "
            f"got {sample_rate_khz}."
        )

    if min_distance_m < profile.range_min_m:
        raise LidarConfigurationError(
            f"min_distance {min_distance_m}m is below the {profile.model}'s "
            f"{profile.range_min_m}m minimum range."
        )

    if max_distance_m > profile.range_max_m:
        raise LidarConfigurationError(
            f"max_distance {max_distance_m}m exceeds the {profile.model}'s "
            f"{profile.range_max_m}m maximum range."
        )

    if min_distance_m >= max_distance_m:
        raise LidarConfigurationError(
            f"min_distance ({min_distance_m}m) must be less than max_distance ({max_distance_m}m)."
        )

    if not (profile.scan_frequency_min_hz <= expected_scan_frequency_hz
            <= profile.scan_frequency_max_hz):
        raise LidarConfigurationError(
            f"expected_scan_frequency_hz {expected_scan_frequency_hz} Hz is outside the "
            f"{profile.model}'s {profile.scan_frequency_min_hz}-"
            f"{profile.scan_frequency_max_hz} Hz range."
        )


class YDLidarX4ProDriver(BaseLidar):
    def __init__(
        self,
        port: str = "auto",
        baudrate: int = X4_PRO.baudrate,
        sample_rate_khz: int = X4_PRO.sample_rate_khz,
        expected_scan_frequency_hz: float = 10.0,
        min_distance_m: float = X4_PRO.range_min_m,
        max_distance_m: float = X4_PRO.range_max_m,
        lidar_type: str = X4_PRO.lidar_type,
        single_channel: bool = X4_PRO.single_channel,
        is_tof: bool = X4_PRO.is_tof,
        frame_id: str = "lidar",
        reversion: bool = True,
        inverted: bool = False,
        angle_offset_deg: float = 0.0,
        auto_reconnect: bool = True,
        scan_timeout_s: float = 2.0,
        reconnect_interval_s: float = 2.0,
        profile: X4ProProfile = X4_PRO,
    ) -> None:
        validate_x4_pro_config(
            baudrate=baudrate,
            lidar_type=lidar_type,
            single_channel=single_channel,
            is_tof=is_tof,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            sample_rate_khz=sample_rate_khz,
            expected_scan_frequency_hz=expected_scan_frequency_hz,
            profile=profile,
        )

        self.configured_port = port
        self.baudrate = baudrate
        self.sample_rate_khz = sample_rate_khz
        # EXPECTED, not commanded. The X4 Pro's rotation speed is set by its
        # motor/hardware; this project does not assume the SDK can change it.
        # Used for: validating the configuration is plausible, health
        # monitoring, and diagnostics (measured vs expected). Never treat a
        # scan as "on time" merely because this value says it should be.
        self.expected_scan_frequency_hz = expected_scan_frequency_hz
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        self.lidar_type = lidar_type
        self.single_channel = single_channel
        self.is_tof = is_tof
        self.frame_id = frame_id
        self.reversion = reversion
        self.inverted = inverted
        self.angle_offset_deg = angle_offset_deg
        self.auto_reconnect = auto_reconnect
        self.scan_timeout_s = scan_timeout_s
        self.reconnect_interval_s = reconnect_interval_s
        self.profile = profile

        self._laser: Any = None
        self._resolved_port: Optional[str] = None
        self._connected = False
        self._scanning = False
        self._error_message: Optional[str] = None

        self._sequence_id = 0
        self._latest_scan: Optional[LidarScanPacket] = None
        self._last_scan_time: float = 0.0
        self._measured_scan_rate_hz: float = 0.0
        self._last_reconnect_attempt: float = 0.0
        self._last_logged_error: Optional[str] = None

    # -- lifecycle -----------------------------------------------------

    def initialize(self) -> None:
        """Steps 1-8 of the bring-up sequence: config is already validated in
        __init__, so this resolves the port, applies the X4 Pro options,
        initializes the SDK, and runs the device health check. start()
        does the turnOn() and the first scans confirm real data.
        """
        if not _HAS_YDLIDAR_SDK:
            self._fail(
                "YDLiDAR SDK not installed (import ydlidar failed). Install YDLidar-SDK "
                "and its python binding, or set simulation_mode: true in config/lidar.yaml "
                "to run against MockLidar instead."
            )
            return

        try:
            ydlidar.os_init()
        except Exception as exc:  # noqa: BLE001
            self._fail(f"ydlidar.os_init() failed: {exc}")
            return

        port = self._resolve_port()
        if port is None:
            self._fail(
                f"No YDLiDAR X4 Pro found. Tried: {', '.join(self._candidate_ports())} "
                f"(no candidate passed SDK initialization at {self.baudrate} baud). "
                f"Check the USB cable/adapter, permissions on the device node, and that "
                f"the LiDAR is powered."
            )
            return

        self._resolved_port = port
        laser = ydlidar.CYdLidar()
        self._apply_sdk_options(laser, port)

        try:
            if not laser.initialize():
                self._fail(f"SDK initialize() failed on {port}: {self._sdk_error(laser)}")
                return
        except Exception as exc:  # noqa: BLE001
            self._fail(f"SDK initialize() raised on {port}: {exc}")
            return

        self._laser = laser
        self._connected = True
        self._error_message = None
        logger.info(
            "YDLiDAR %s initialized on %s @ %d baud (type=%s, single_channel=%s, is_tof=%s, "
            "sample_rate=%dkHz, expected %.1f Hz, range %.2f-%.2fm)",
            self.profile.model, port, self.baudrate, self.lidar_type, self.single_channel,
            self.is_tof, self.sample_rate_khz, self.expected_scan_frequency_hz,
            self.min_distance_m, self.max_distance_m,
        )

    def _apply_sdk_options(self, laser: Any, port: str) -> None:
        """Steps 3-6: the X4 Pro's fixed profile, expressed in SDK terms.

        Note LidarPropIntenstiy — that spelling is the SDK's own, not a typo
        here.

        LidarPropScanFrequency is passed because it is the documented option
        for expressing a desired rotation rate, but it is treated as a HINT,
        not a guarantee: on this device the rate is governed by the motor,
        and the SDK may ignore the value entirely. Nothing in this project
        assumes the request took effect — health and diagnostics compare the
        MEASURED rate (LidarStatus.scan_rate_hz) against the expectation
        (LidarStatus.expected_scan_rate_hz).
        """
        laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
        laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, int(self.baudrate))
        laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
        laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        laser.setlidaropt(ydlidar.LidarPropScanFrequency, float(self.expected_scan_frequency_hz))
        laser.setlidaropt(ydlidar.LidarPropSampleRate, int(self.sample_rate_khz))
        laser.setlidaropt(ydlidar.LidarPropSingleChannel, bool(self.single_channel))
        laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
        laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
        laser.setlidaropt(ydlidar.LidarPropMaxRange, float(self.max_distance_m))
        laser.setlidaropt(ydlidar.LidarPropMinRange, float(self.min_distance_m))
        laser.setlidaropt(ydlidar.LidarPropIntenstiy, False)
        laser.setlidaropt(ydlidar.LidarPropSupportMotorDtrCtrl, True)
        laser.setlidaropt(ydlidar.LidarPropReversion, bool(self.reversion))
        laser.setlidaropt(ydlidar.LidarPropInverted, bool(self.inverted))
        laser.setlidaropt(ydlidar.LidarPropAutoReconnect, bool(self.auto_reconnect))

    # -- port resolution -----------------------------------------------------

    def _candidate_ports(self) -> list[str]:
        """SDK-enumerated ports first (it knows the adapter's VID/PID), then
        the documented device paths that actually exist on this machine.

        A manual `port:` in config/lidar.yaml short-circuits all of this.
        """
        if self.configured_port and self.configured_port != "auto":
            return [self.configured_port]

        candidates: list[str] = []

        if _HAS_YDLIDAR_SDK:
            try:
                for _name, path in ydlidar.lidarPortList().items():
                    if path not in candidates:
                        candidates.append(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ydlidar.lidarPortList() failed (%s); falling back to the "
                               "documented device paths.", exc)

        for path in DEFAULT_PORT_CANDIDATES:
            if path not in candidates and os.path.exists(path):
                candidates.append(path)

        return candidates

    def _resolve_port(self) -> Optional[str]:
        """Step 2. Never connects to an arbitrary serial device: each
        candidate must pass a real SDK initialize() (which talks to the
        device and reads its health/device info) before it is accepted.
        A USB-serial adapter for something else on /dev/ttyUSB0 therefore
        gets rejected rather than driven as if it were a LiDAR.
        """
        candidates = self._candidate_ports()
        if not candidates:
            logger.error("No candidate LiDAR serial ports exist on this machine.")
            return None

        # An explicit port is still probed, so a wrong/stale path in config
        # surfaces as a clear error instead of a silent no-data condition.
        for port in candidates:
            if self._verify_port(port):
                logger.info("YDLiDAR %s verified on %s", self.profile.model, port)
                return port
            logger.debug("Candidate port %s did not respond as a YDLiDAR %s",
                         port, self.profile.model)

        return None

    def _verify_port(self, port: str) -> bool:
        """Probe one candidate with a throwaway SDK handle, then release it
        so initialize() can open the device cleanly.
        """
        if not _HAS_YDLIDAR_SDK:
            return False

        probe = None
        try:
            probe = ydlidar.CYdLidar()
            self._apply_sdk_options(probe, port)
            return bool(probe.initialize())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Probe of %s raised: %s", port, exc)
            return False
        finally:
            if probe is not None:
                try:
                    probe.turnOff()
                    probe.disconnecting()
                except Exception:  # noqa: BLE001
                    pass

    # -- scanning -----------------------------------------------------

    def start(self) -> None:
        """Step 9. No-op (staying unhealthy) if initialize() failed, so a
        missing LiDAR never takes down the caller."""
        if self._laser is None:
            logger.error("YDLiDAR %s cannot start: %s", self.profile.model,
                         self._error_message or "not initialized")
            return

        try:
            if not self._laser.turnOn():
                self._fail(f"turnOn() failed: {self._sdk_error(self._laser)}")
                return
        except Exception as exc:  # noqa: BLE001
            self._fail(f"turnOn() raised: {exc}")
            return

        self._scanning = True
        self._error_message = None
        logger.info("YDLiDAR %s scanning on %s", self.profile.model, self._resolved_port)

    def get_scan(self, timeout_s: float = 1.0) -> Optional[LidarScanPacket]:
        """Step 10. One blocking SDK read (doProcessSimple returns a full
        rotation), converted into the project's own units/convention.

        Returns None — never raises — on any device error, and applies the
        reconnect policy so a USB disconnect mid-run is recoverable without
        restarting the process.
        """
        if not self._scanning or self._laser is None:
            self._maybe_reconnect()
            return None

        scan_holder = ydlidar.LaserScan()
        try:
            ok = self._laser.doProcessSimple(scan_holder)
        except Exception as exc:  # noqa: BLE001
            self._fail(f"doProcessSimple() raised (device disconnected?): {exc}")
            self._maybe_reconnect()
            return None

        if not ok:
            self._note_scan_failure(f"doProcessSimple() returned false: {self._sdk_error(self._laser)}")
            self._maybe_reconnect()
            return None

        packet = self._convert_scan(scan_holder)
        if packet is None:
            return None

        self._latest_scan = packet
        self._update_scan_rate(packet.timestamp)
        self._last_scan_time = packet.timestamp
        self._error_message = None
        # Real data is flowing again: reset the log-dedup memory so the next
        # distinct failure is reported loudly rather than suppressed as a
        # "repeat" of something that has since been resolved.
        self._last_logged_error = None
        return packet

    def _convert_scan(self, scan_holder: Any) -> Optional[LidarScanPacket]:
        """SDK LaserScan -> LidarScanPacket.

        The SDK reports angles in RADIANS and ranges in METERS; this project
        stores signed DEGREES in [-180, 180) with 0 = forward (see
        common/types.py). `reversion`/`inverted` are applied by the SDK
        itself via its own options, so only the project's extra
        angle_offset_deg calibration is applied here — doing it in both
        places would double-count.
        """
        try:
            raw_points = list(scan_holder.points)
        except Exception as exc:  # noqa: BLE001
            self._note_scan_failure(f"could not read scan points: {exc}")
            return None

        points: list[LidarPoint] = []
        timestamp = self._scan_timestamp(scan_holder)

        for raw in raw_points:
            angle_deg = math.degrees(float(raw.angle)) + self.angle_offset_deg
            angle_deg = ((angle_deg + 180.0) % 360.0) - 180.0
            intensity = float(getattr(raw, "intensity", 0.0) or 0.0)
            points.append(LidarPoint(
                angle_deg=angle_deg,
                distance_m=float(raw.range),
                intensity=intensity,
                timestamp=timestamp,
            ))

        scan_duration_s = self._scan_duration(scan_holder)
        self._sequence_id += 1

        return LidarScanPacket(
            points=points,
            timestamp=timestamp,
            sequence_id=self._sequence_id,
            frame_id=self.frame_id,
            source=f"ydlidar_{self.profile.model.lower()}",
            scan_duration_s=scan_duration_s,
            scan_frequency_hz=(1.0 / scan_duration_s) if scan_duration_s > 0 else 0.0,
            valid_points=len(points),   # ScanProcessor refines this after range filtering
            invalid_points=0,
        )

    @staticmethod
    def _scan_timestamp(scan_holder: Any) -> float:
        """SDK stamps are nanoseconds since epoch; fall back to host time if
        the field is missing or implausible (some SDK builds leave it 0).
        """
        stamp_ns = getattr(scan_holder, "stamp", 0)
        try:
            stamp_s = float(stamp_ns) / 1e9
        except (TypeError, ValueError):
            return time.time()
        return stamp_s if stamp_s > 1e6 else time.time()

    @staticmethod
    def _scan_duration(scan_holder: Any) -> float:
        config = getattr(scan_holder, "config", None)
        try:
            return float(getattr(config, "scan_time", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def get_latest_scan(self) -> Optional[LidarScanPacket]:
        return self._latest_scan

    def _update_scan_rate(self, timestamp: float) -> None:
        if self._last_scan_time > 0.0:
            delta = timestamp - self._last_scan_time
            if delta > 0:
                instantaneous = 1.0 / delta
                # Light smoothing so one late scan doesn't make the reported
                # rate jump around in the diagnostic view.
                self._measured_scan_rate_hz = (
                    instantaneous if self._measured_scan_rate_hz == 0.0
                    else 0.7 * self._measured_scan_rate_hz + 0.3 * instantaneous
                )

    # -- health / recovery -----------------------------------------------------

    def is_healthy(self) -> bool:
        if not self._connected or not self._scanning:
            return False
        if self._last_scan_time == 0.0:
            return False
        return (time.time() - self._last_scan_time) <= self.scan_timeout_s

    def get_status(self) -> LidarStatus:
        return LidarStatus(
            connected=self._connected,
            scanning=self._scanning,
            healthy=self.is_healthy(),
            port=self._resolved_port,
            baudrate=self.baudrate,
            model=self.profile.model,
            last_scan_timestamp=self._last_scan_time or None,
            scan_rate_hz=round(self._measured_scan_rate_hz, 2),
            expected_scan_rate_hz=self.expected_scan_frequency_hz,
            error_message=self._error_message,
            simulated=False,
        )

    def _fail(self, message: str) -> None:
        """Structured failure: record it, mark unhealthy, publish nothing.

        A persistent fault (unplugged sensor, missing SDK) would otherwise
        re-log the same error on every reconnect attempt forever. The first
        occurrence is an ERROR; identical repeats drop to DEBUG so the log
        stays readable and a *changed* failure still stands out.
        """
        repeated = (message == self._last_logged_error)
        self._error_message = message
        self._connected = False
        self._scanning = False

        log = logger.debug if repeated else logger.error
        log("YDLiDAR %s error | port=%s baudrate=%s | %s%s",
            self.profile.model, self._resolved_port or self.configured_port,
            self.baudrate, message, " (repeated)" if repeated else "")
        self._last_logged_error = message

    def _note_scan_failure(self, message: str) -> None:
        """A single bad read — worth recording, but not necessarily a
        disconnect. is_healthy() still governs via scan_timeout_s.
        """
        self._error_message = message
        logger.warning("YDLiDAR %s scan failure: %s", self.profile.model, message)

    def _maybe_reconnect(self) -> None:
        if not self.auto_reconnect:
            return
        now = time.time()
        if (now - self._last_reconnect_attempt) < self.reconnect_interval_s:
            return
        self._last_reconnect_attempt = now

        # Same dedup rationale as _fail(): the first attempt after a healthy
        # period is worth an INFO line, a stuck retry loop is not.
        if self._last_logged_error is None:
            logger.info("YDLiDAR %s attempting reconnect...", self.profile.model)
        else:
            logger.debug("YDLiDAR %s retrying reconnect...", self.profile.model)
        self._release_sdk_handle()
        self._connected = False
        self._scanning = False
        self._measured_scan_rate_hz = 0.0
        self._last_scan_time = 0.0

        self.initialize()
        if self._laser is not None:
            self.start()

    @staticmethod
    def _sdk_error(laser: Any) -> str:
        try:
            return str(laser.DescribeError())
        except Exception:  # noqa: BLE001
            return "no SDK error description available"

    # -- teardown -----------------------------------------------------

    def stop(self) -> None:
        self._scanning = False
        if self._laser is not None:
            try:
                self._laser.turnOff()
            except Exception as exc:  # noqa: BLE001
                logger.warning("YDLiDAR %s turnOff() failed: %s", self.profile.model, exc)

    def shutdown(self) -> None:
        self.stop()
        self._release_sdk_handle()
        self._connected = False

    def _release_sdk_handle(self) -> None:
        if self._laser is None:
            return
        try:
            self._laser.disconnecting()
        except Exception as exc:  # noqa: BLE001
            logger.warning("YDLiDAR %s disconnecting() failed: %s", self.profile.model, exc)
        finally:
            self._laser = None


def build_ydlidar_x4_pro_from_config(cfg: dict) -> YDLidarX4ProDriver:
    l_cfg = cfg["lidar"]
    return YDLidarX4ProDriver(
        port=l_cfg.get("port", "auto"),
        baudrate=l_cfg.get("baudrate", X4_PRO.baudrate),
        sample_rate_khz=l_cfg.get("sample_rate", X4_PRO.sample_rate_khz),
        expected_scan_frequency_hz=l_cfg.get("expected_scan_frequency_hz", 10.0),
        min_distance_m=l_cfg.get("min_distance", X4_PRO.range_min_m),
        max_distance_m=l_cfg.get("max_distance", X4_PRO.range_max_m),
        lidar_type=l_cfg.get("lidar_type", X4_PRO.lidar_type),
        single_channel=l_cfg.get("single_channel", X4_PRO.single_channel),
        is_tof=l_cfg.get("is_tof", X4_PRO.is_tof),
        frame_id=l_cfg.get("frame_id", "lidar"),
        reversion=l_cfg.get("reversion", True),
        inverted=l_cfg.get("inverted", False),
        angle_offset_deg=l_cfg.get("angle_offset", l_cfg.get("angle_offset_deg", 0.0)),
        auto_reconnect=l_cfg.get("auto_reconnect", True),
        scan_timeout_s=l_cfg.get("scan_timeout", 2.0),
    )
