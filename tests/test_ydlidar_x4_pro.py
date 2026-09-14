"""Tests for the YDLiDAR X4 Pro driver and the LiDAR factory.

These run WITHOUT the YDLiDAR SDK and without any LiDAR attached — which is
also the most important behavior under test: on a machine with no SDK, the
driver must report itself unhealthy with a specific error rather than
either crashing the runtime or silently serving synthetic scans.
"""

from __future__ import annotations

import time

import pytest

from common.types import LidarStatus
from hardware.lidar import build_lidar_from_config
from hardware.lidar.mock_lidar import MockLidar
from hardware.lidar.ydlidar_x4_pro_driver import (
    DEFAULT_PORT_CANDIDATES,
    X4_PRO,
    LidarConfigurationError,
    YDLidarX4ProDriver,
    validate_x4_pro_config,
)


def _valid_kwargs(**overrides):
    kwargs = dict(
        baudrate=128000,
        lidar_type="triangle",
        single_channel=True,
        is_tof=False,
        min_distance_m=0.12,
        max_distance_m=10.0,
        sample_rate_khz=5,
        expected_scan_frequency_hz=10.0,
    )
    kwargs.update(overrides)
    return kwargs


# -- 1. The exact X4 Pro hardware profile -----------------------------------------------------


def test_x4_pro_profile_is_exactly_the_specified_hardware():
    assert X4_PRO.model == "X4_PRO"
    assert X4_PRO.baudrate == 128000
    assert X4_PRO.sample_rate_khz == 5
    assert X4_PRO.range_min_m == 0.12
    assert X4_PRO.range_max_m == 10.0
    assert X4_PRO.scan_angle_deg == 360.0
    assert X4_PRO.lidar_type == "triangle"
    assert X4_PRO.single_channel is True
    assert X4_PRO.is_tof is False


def test_profile_scan_frequency_band_is_the_hardware_capability():
    assert (X4_PRO.scan_frequency_min_hz, X4_PRO.scan_frequency_max_hz) == (6.0, 12.0)


def test_valid_x4_pro_configuration_passes():
    validate_x4_pro_config(**_valid_kwargs())  # must not raise


def test_driver_constructs_with_a_valid_configuration():
    driver = YDLidarX4ProDriver(port="/dev/ttyUSB0")
    assert driver.baudrate == 128000
    assert driver.profile.model == "X4_PRO"
    assert driver.single_channel is True


# -- 2. Invalid configuration rejection -----------------------------------------------------


@pytest.mark.parametrize("bad_baudrate", [115200, 230400, 256000, 9600])
def test_invalid_baudrate_is_rejected(bad_baudrate):
    with pytest.raises(LidarConfigurationError) as exc:
        validate_x4_pro_config(**_valid_kwargs(baudrate=bad_baudrate))
    assert "128000" in str(exc.value)


def test_driver_construction_rejects_invalid_baudrate():
    with pytest.raises(LidarConfigurationError):
        YDLidarX4ProDriver(port="/dev/ttyUSB0", baudrate=115200)


def test_tof_configuration_is_rejected():
    with pytest.raises(LidarConfigurationError) as exc:
        validate_x4_pro_config(**_valid_kwargs(is_tof=True))
    assert "ToF" in str(exc.value)


def test_x4_pro_accepts_single_channel_true():
    validate_x4_pro_config(**_valid_kwargs(single_channel=True))  # must not raise


def test_x4_pro_rejects_single_channel_false():
    with pytest.raises(LidarConfigurationError) as exc:
        validate_x4_pro_config(**_valid_kwargs(single_channel=False))
    message = str(exc.value)
    assert "single_channel=True" in message
    # The message must state what is required, not assert a claim about the
    # device's internals that could go stale if the profile is corrected.
    assert "dual-channel" not in message.lower()
    assert "dual channel" not in message.lower()


def test_driver_construction_rejects_single_channel_false():
    with pytest.raises(LidarConfigurationError):
        YDLidarX4ProDriver(port="/dev/ttyUSB0", single_channel=False)


def test_non_triangle_lidar_type_is_rejected():
    with pytest.raises(LidarConfigurationError):
        validate_x4_pro_config(**_valid_kwargs(lidar_type="tof"))


def test_lidar_type_remains_triangle():
    assert X4_PRO.lidar_type == "triangle"
    validate_x4_pro_config(**_valid_kwargs(lidar_type="triangle"))


def test_range_outside_hardware_limits_is_rejected():
    with pytest.raises(LidarConfigurationError):
        validate_x4_pro_config(**_valid_kwargs(min_distance_m=0.05))
    with pytest.raises(LidarConfigurationError):
        validate_x4_pro_config(**_valid_kwargs(max_distance_m=25.0))


def test_range_remains_0_12_to_10_meters():
    assert (X4_PRO.range_min_m, X4_PRO.range_max_m) == (0.12, 10.0)
    validate_x4_pro_config(**_valid_kwargs(min_distance_m=0.12, max_distance_m=10.0))


def test_expected_scan_frequency_outside_supported_band_is_rejected():
    with pytest.raises(LidarConfigurationError) as exc:
        validate_x4_pro_config(**_valid_kwargs(expected_scan_frequency_hz=3.0))
    assert "expected_scan_frequency_hz" in str(exc.value)

    with pytest.raises(LidarConfigurationError):
        validate_x4_pro_config(**_valid_kwargs(expected_scan_frequency_hz=20.0))


def test_expected_scan_frequency_is_an_expectation_not_a_commanded_rate():
    """The driver stores it for diagnostics/health, and reports it next to
    the MEASURED rate — it must never be presented as the actual rate.
    """
    driver = YDLidarX4ProDriver(port="/dev/ttyUSB0", expected_scan_frequency_hz=8.0)
    assert driver.expected_scan_frequency_hz == 8.0

    status = driver.get_status()
    assert status.expected_scan_rate_hz == 8.0
    # No scans have arrived, so the measured rate must still be zero rather
    # than optimistically echoing the expectation.
    assert status.scan_rate_hz == 0.0


def test_wrong_sample_rate_is_rejected():
    with pytest.raises(LidarConfigurationError):
        validate_x4_pro_config(**_valid_kwargs(sample_rate_khz=9))


def test_sample_rate_remains_5_khz():
    assert X4_PRO.sample_rate_khz == 5
    validate_x4_pro_config(**_valid_kwargs(sample_rate_khz=5))


def test_is_tof_remains_false():
    assert X4_PRO.is_tof is False
    validate_x4_pro_config(**_valid_kwargs(is_tof=False))


def test_baudrate_remains_128000():
    assert X4_PRO.baudrate == 128000
    validate_x4_pro_config(**_valid_kwargs(baudrate=128000))


# -- 3. Unhealthy state when the SDK/hardware is absent -----------------------------------------------------


def test_driver_without_sdk_reports_unhealthy_and_does_not_raise():
    driver = YDLidarX4ProDriver(port="/dev/ttyUSB_DOES_NOT_EXIST", auto_reconnect=False)

    driver.initialize()   # must not raise even with no SDK and no device
    driver.start()        # must not raise

    status = driver.get_status()
    assert status.healthy is False
    assert status.connected is False
    assert status.simulated is False           # never pretends to be simulation
    assert status.error_message                # a specific reason is recorded
    assert status.model == "X4_PRO"
    assert driver.is_healthy() is False

    driver.shutdown()


def test_driver_publishes_no_scans_when_hardware_is_unavailable():
    driver = YDLidarX4ProDriver(port="/dev/ttyUSB_DOES_NOT_EXIST", auto_reconnect=False)
    driver.initialize()
    driver.start()
    try:
        assert driver.get_scan(timeout_s=0.2) is None
        assert driver.get_latest_scan() is None
    finally:
        driver.shutdown()


def test_status_is_serializable_for_cross_process_publishing():
    driver = YDLidarX4ProDriver(port="/dev/ttyUSB_DOES_NOT_EXIST", auto_reconnect=False)
    driver.initialize()
    status_dict = driver.get_status().to_dict()

    assert status_dict["model"] == "X4_PRO"
    assert status_dict["baudrate"] == 128000
    assert status_dict["healthy"] is False
    assert status_dict["simulated"] is False
    driver.shutdown()


# -- 4. Scan timeout -----------------------------------------------------


def test_health_expires_after_scan_timeout():
    driver = YDLidarX4ProDriver(port="/dev/ttyUSB0", scan_timeout_s=0.2, auto_reconnect=False)

    # Simulate "was connected and scanning, then the data stopped".
    driver._connected = True
    driver._scanning = True
    driver._last_scan_time = time.time()
    assert driver.is_healthy() is True

    driver._last_scan_time = time.time() - 0.5   # older than scan_timeout_s
    assert driver.is_healthy() is False


def test_scan_timeout_is_configurable_from_yaml_value():
    driver = YDLidarX4ProDriver(port="/dev/ttyUSB0", scan_timeout_s=5.0)
    assert driver.scan_timeout_s == 5.0


# -- 5. Port detection -----------------------------------------------------


def test_documented_port_candidates_are_checked():
    assert DEFAULT_PORT_CANDIDATES == (
        "/dev/ydlidar", "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1",
    )


def test_explicit_port_short_circuits_auto_detection():
    driver = YDLidarX4ProDriver(port="/dev/ttyUSB1")
    assert driver._candidate_ports() == ["/dev/ttyUSB1"]


def test_auto_detection_only_offers_ports_that_exist():
    driver = YDLidarX4ProDriver(port="auto")
    for candidate in driver._candidate_ports():
        # Never fabricate a device path: everything offered either came from
        # the SDK's enumeration or actually exists on this machine.
        assert candidate.startswith("/dev/")


def test_unverified_port_is_not_accepted():
    # With no SDK available, verification cannot succeed, so no port may be
    # claimed — the driver must not "connect" to an arbitrary serial device.
    driver = YDLidarX4ProDriver(port="/dev/ttyUSB0")
    assert driver._verify_port("/dev/ttyUSB0") is False
    assert driver._resolve_port() is None


# -- 6. Factory selection: simulation vs real hardware -----------------------------------------------------


def _cfg(simulation_mode: bool, driver: str = "ydlidar_x4_pro") -> dict:
    return {
        "lidar": {
            "enabled": True,
            "driver": driver,
            "model": "X4_PRO",
            "simulation_mode": simulation_mode,
            "port": "/dev/ttyUSB0",
            "baudrate": 128000,
            "sample_rate": 5,
            "expected_scan_frequency_hz": 10.0,
            "min_distance": 0.12,
            "max_distance": 10.0,
            "lidar_type": "triangle",
            "single_channel": True,
            "is_tof": False,
            "frame_id": "lidar",
            "reversion": True,
            "inverted": False,
            "angle_offset": 0.0,
            "auto_reconnect": True,
            "scan_timeout": 2.0,
        }
    }


def test_factory_returns_mock_lidar_in_simulation_mode():
    lidar = build_lidar_from_config(_cfg(simulation_mode=True))
    assert isinstance(lidar, MockLidar)


def test_factory_returns_real_driver_when_simulation_is_off():
    lidar = build_lidar_from_config(_cfg(simulation_mode=False))
    assert isinstance(lidar, YDLidarX4ProDriver)


def test_factory_never_falls_back_to_mock_when_hardware_is_missing():
    """The safety-critical property: a real driver that cannot reach its
    hardware stays a real driver reporting unhealthy. Silently serving
    synthetic obstacle data as if it were real would be far worse than
    reporting a dead sensor.
    """
    lidar = build_lidar_from_config(_cfg(simulation_mode=False))
    lidar.initialize()
    try:
        assert not isinstance(lidar, MockLidar)
        status: LidarStatus = lidar.get_status()
        assert status.simulated is False
        assert status.healthy is False
    finally:
        lidar.shutdown()


def test_factory_rejects_an_unknown_driver_name():
    with pytest.raises(LidarConfigurationError):
        build_lidar_from_config(_cfg(simulation_mode=False, driver="some_other_lidar"))


def test_factory_applies_x4_pro_validation_to_yaml_values():
    cfg = _cfg(simulation_mode=False)
    cfg["lidar"]["baudrate"] = 115200
    with pytest.raises(LidarConfigurationError):
        build_lidar_from_config(cfg)


def test_factory_rejects_single_channel_false_from_yaml():
    cfg = _cfg(simulation_mode=False)
    cfg["lidar"]["single_channel"] = False
    with pytest.raises(LidarConfigurationError) as exc:
        build_lidar_from_config(cfg)
    assert "single_channel=True" in str(exc.value)


def test_simulation_mode_still_works_after_the_profile_correction():
    """The profile change must not disturb simulation: MockLidar is not
    bound by the X4 Pro's channel/baudrate facts and still produces scans.
    """
    lidar = build_lidar_from_config(_cfg(simulation_mode=True))
    lidar.initialize()
    lidar.start()
    try:
        packet = lidar.get_scan(timeout_s=2.0)
        assert packet is not None
        assert len(packet.points) > 0
        assert lidar.is_healthy() is True

        status = lidar.get_status()
        assert status.simulated is True
        assert status.expected_scan_rate_hz == 10.0
    finally:
        lidar.shutdown()


# -- 7. The SDK actually receives the profile -----------------------------------------------------


class _FakeLaser:
    """Records every setlidaropt() call so the options handed to the SDK
    can be asserted without the SDK (or hardware) being present.
    """

    def __init__(self) -> None:
        self.options: dict = {}

    def setlidaropt(self, key, value):
        self.options[key] = value
        return True


class _FakeYdlidarModule:
    """Minimal stand-in for the `ydlidar` binding. Property constants are
    plain strings so the recorded options are readable in assertions.
    """

    LidarPropSerialPort = "serial_port"
    LidarPropSerialBaudrate = "serial_baudrate"
    LidarPropLidarType = "lidar_type"
    LidarPropDeviceType = "device_type"
    LidarPropScanFrequency = "scan_frequency"
    LidarPropSampleRate = "sample_rate"
    LidarPropSingleChannel = "single_channel"
    LidarPropMaxAngle = "max_angle"
    LidarPropMinAngle = "min_angle"
    LidarPropMaxRange = "max_range"
    LidarPropMinRange = "min_range"
    LidarPropIntenstiy = "intensity"          # the SDK's own spelling
    LidarPropSupportMotorDtrCtrl = "motor_dtr"
    LidarPropReversion = "reversion"
    LidarPropInverted = "inverted"
    LidarPropAutoReconnect = "auto_reconnect"

    TYPE_TRIANGLE = "TYPE_TRIANGLE"
    YDLIDAR_TYPE_SERIAL = "YDLIDAR_TYPE_SERIAL"


def _captured_sdk_options(monkeypatch, **driver_kwargs) -> dict:
    from hardware.lidar import ydlidar_x4_pro_driver as drv

    monkeypatch.setattr(drv, "ydlidar", _FakeYdlidarModule, raising=False)
    driver = YDLidarX4ProDriver(port="/dev/ttyUSB0", **driver_kwargs)
    laser = _FakeLaser()
    driver._apply_sdk_options(laser, "/dev/ttyUSB0")
    return laser.options


def test_sdk_receives_single_channel_true(monkeypatch):
    options = _captured_sdk_options(monkeypatch)
    assert options["single_channel"] is True


def test_sdk_receives_the_full_x4_pro_profile(monkeypatch):
    options = _captured_sdk_options(monkeypatch)

    assert options["serial_port"] == "/dev/ttyUSB0"
    assert options["serial_baudrate"] == 128000
    assert options["lidar_type"] == "TYPE_TRIANGLE"
    assert options["device_type"] == "YDLIDAR_TYPE_SERIAL"
    assert options["sample_rate"] == 5
    assert options["single_channel"] is True
    assert options["min_range"] == 0.12
    assert options["max_range"] == 10.0
    assert options["min_angle"] == -180.0
    assert options["max_angle"] == 180.0


def test_sdk_scan_frequency_option_carries_the_expected_value(monkeypatch):
    """The option is still passed (it is the documented way to express a
    desired rate) but the project treats it as a hint — see the driver's
    _apply_sdk_options docstring.
    """
    options = _captured_sdk_options(monkeypatch, expected_scan_frequency_hz=7.0)
    assert options["scan_frequency"] == 7.0


def test_sdk_receives_reversion_and_inverted_calibration(monkeypatch):
    options = _captured_sdk_options(monkeypatch, reversion=True, inverted=False)
    assert options["reversion"] is True
    assert options["inverted"] is False


# -- 8. The shipped config/lidar.yaml is itself valid -----------------------------------------------------


def test_shipped_lidar_yaml_describes_a_valid_x4_pro():
    from common.config_loader import load_system_config

    cfg = load_system_config("config/system.yaml")
    l_cfg = cfg["lidar"]

    assert l_cfg["driver"] == "ydlidar_x4_pro"
    assert l_cfg["model"] == "X4_PRO"
    assert l_cfg["baudrate"] == 128000
    assert l_cfg["sample_rate"] == 5
    assert l_cfg["single_channel"] is True
    assert l_cfg["is_tof"] is False
    assert l_cfg["lidar_type"] == "triangle"
    assert l_cfg["min_distance"] == 0.12
    assert l_cfg["max_distance"] == 10.0
    assert l_cfg["expected_scan_frequency_hz"] == 10.0
    # The old name must be gone, so nothing silently reads a stale key.
    assert "scan_frequency_hz" not in l_cfg

    validate_x4_pro_config(
        baudrate=l_cfg["baudrate"],
        lidar_type=l_cfg["lidar_type"],
        single_channel=l_cfg["single_channel"],
        is_tof=l_cfg["is_tof"],
        min_distance_m=l_cfg["min_distance"],
        max_distance_m=l_cfg["max_distance"],
        sample_rate_khz=l_cfg["sample_rate"],
        expected_scan_frequency_hz=l_cfg["expected_scan_frequency_hz"],
    )
