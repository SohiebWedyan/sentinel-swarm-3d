"""Unit tests for the LiDAR stack: scan filtering, obstacle-sector math
(including angle wraparound at REAR and the safety banding), and MockLidar.

All of these run with no LiDAR hardware and no YDLiDAR SDK installed.
Driver-specific tests live in tests/test_ydlidar_x4_pro.py.
"""

from __future__ import annotations

import math

from common.types import (
    LidarPoint,
    LidarSafetyLevel,
    LidarScanPacket,
    LidarSectorName,
)
from hardware.lidar.mock_lidar import MockLidar
from hardware.lidar.obstacle_sectors import (
    DEFAULT_SECTOR_RANGES_DEG,
    ObstacleSectorAnalyzer,
    SafetyThresholds,
)
from hardware.lidar.scan_processor import ScanProcessor


def _packet(pairs) -> LidarScanPacket:
    return LidarScanPacket(points=[LidarPoint(angle_deg=a, distance_m=d) for a, d in pairs])


# -- ScanProcessor: range filtering (X4 Pro window) -----------------------------------------------------


def test_scan_processor_drops_out_of_range_and_invalid_points():
    processor = ScanProcessor(min_distance_m=0.12, max_distance_m=10.0)
    scan = _packet([
        (0, 2.0),
        (10, 0.05),          # below the X4 Pro's 0.12m minimum
        (20, 12.0),          # beyond the X4 Pro's 10.0m maximum
        (30, math.nan),
        (40, math.inf),
        (50, 2.1),
        (60, 0.0),           # zero return = no measurement
    ])

    filtered = processor.process(scan)

    assert sorted(p.distance_m for p in filtered.points) == [2.0, 2.1]


def test_scan_processor_records_valid_and_invalid_counts():
    processor = ScanProcessor(min_distance_m=0.12, max_distance_m=10.0)
    scan = _packet([(0, 2.0), (10, 0.05), (20, 50.0), (30, 3.0)])

    filtered = processor.process(scan)

    assert filtered.valid_points == 2
    assert filtered.invalid_points == 2


def test_scan_processor_accepts_exact_x4_pro_range_boundaries():
    processor = ScanProcessor(min_distance_m=0.12, max_distance_m=10.0)
    scan = _packet([(0, 0.12), (10, 10.0)])

    filtered = processor.process(scan)

    assert filtered.valid_points == 2


def test_scan_processor_preserves_packet_metadata():
    processor = ScanProcessor(min_distance_m=0.12, max_distance_m=10.0)
    scan = LidarScanPacket(
        points=[LidarPoint(angle_deg=0.0, distance_m=1.0)],
        sequence_id=42,
        frame_id="lidar",
        source="ydlidar_x4_pro",
        scan_duration_s=0.1,
        scan_frequency_hz=10.0,
    )

    filtered = processor.process(scan)

    assert filtered.sequence_id == 42
    assert filtered.source == "ydlidar_x4_pro"
    assert filtered.scan_frequency_hz == 10.0


def test_scan_processor_normalizes_zero_to_360_angles_into_signed_range():
    processor = ScanProcessor(min_distance_m=0.12, max_distance_m=10.0)
    scan = _packet([(350.0, 1.0), (10.0, 1.0), (200.0, 1.0)])

    filtered = processor.process(scan)

    assert all(-180.0 <= p.angle_deg < 180.0 for p in filtered.points)
    # 350 deg is 10 deg to the right of forward, i.e. -10 in signed form.
    assert any(abs(p.angle_deg + 10.0) < 1e-9 for p in filtered.points)


# -- ScanProcessor: noise filtering must not eat real obstacles -----------------------------------------------------


def test_scan_processor_rejects_isolated_spike_between_consistent_neighbors():
    pairs = [(float(i), 2.0) for i in range(0, 360, 10)]
    pairs[5] = (pairs[5][0], 6.0)  # lone spike, both neighbors read ~2.0
    processor = ScanProcessor(min_distance_m=0.12, max_distance_m=10.0, max_neighbor_jump_m=1.0)

    filtered = processor.process(_packet(pairs))

    assert all(p.distance_m == 2.0 for p in filtered.points)
    assert filtered.invalid_points == 1


def test_scan_processor_keeps_a_real_obstacle_edge_where_only_one_side_jumps():
    pairs = [(float(i), 2.0) for i in range(0, 180, 10)] + \
            [(float(i), 0.5) for i in range(180, 360, 10)]
    processor = ScanProcessor(min_distance_m=0.12, max_distance_m=10.0, max_neighbor_jump_m=1.0)

    filtered = processor.process(_packet(pairs))

    assert filtered.invalid_points == 0


def test_scan_processor_noise_filter_can_be_disabled():
    pairs = [(float(i), 2.0) for i in range(0, 360, 10)]
    pairs[5] = (pairs[5][0], 6.0)
    processor = ScanProcessor(min_distance_m=0.12, max_distance_m=10.0,
                              max_neighbor_jump_m=1.0, enable_noise_filter=False)

    filtered = processor.process(_packet(pairs))

    assert filtered.invalid_points == 0
    assert any(p.distance_m == 6.0 for p in filtered.points)


# -- ObstacleSectorAnalyzer -----------------------------------------------------


def test_front_sector_detects_close_obstacle():
    analyzer = ObstacleSectorAnalyzer(obstacle_distance_m=0.8, min_points_for_valid_sector=1)
    sectors = analyzer.analyze(_packet([(0.0, 0.5), (5.0, 0.6), (-5.0, 0.55)]))

    front = sectors[LidarSectorName.FRONT]
    assert front.obstacle_detected is True
    assert front.min_distance_m == 0.5
    assert front.median_distance_m == 0.55
    assert front.num_valid_points == 3


def test_sector_reports_no_obstacle_when_clear():
    analyzer = ObstacleSectorAnalyzer(obstacle_distance_m=0.8, min_points_for_valid_sector=1)
    sectors = analyzer.analyze(_packet([(0.0, 3.0)]))

    assert sectors[LidarSectorName.FRONT].obstacle_detected is False


def test_sector_below_min_points_is_invalid_not_clear():
    analyzer = ObstacleSectorAnalyzer(obstacle_distance_m=0.8, min_points_for_valid_sector=5)
    sectors = analyzer.analyze(_packet([(0.0, 0.3)]))

    front = sectors[LidarSectorName.FRONT]
    assert front.min_distance_m is None
    assert front.obstacle_detected is False


def test_rear_sector_wraps_across_plus_minus_180():
    analyzer = ObstacleSectorAnalyzer(obstacle_distance_m=0.8, min_points_for_valid_sector=1)
    sectors = analyzer.analyze(_packet([(170.0, 1.0), (-170.0, 1.2)]))

    assert sectors[LidarSectorName.REAR].num_valid_points == 2


def test_all_four_clearance_helpers():
    analyzer = ObstacleSectorAnalyzer(obstacle_distance_m=0.8, min_points_for_valid_sector=1)
    sectors = analyzer.analyze(_packet([
        (0.0, 1.5),      # front
        (90.0, 2.5),     # left
        (-90.0, 0.9),    # right
        (180.0, 3.3),    # rear
    ]))

    assert analyzer.get_front_clearance(sectors) == 1.5
    assert analyzer.get_left_clearance(sectors) == 2.5
    assert analyzer.get_right_clearance(sectors) == 0.9
    assert analyzer.get_rear_clearance(sectors) == 3.3


def test_default_sector_ranges_cover_boundary_angles():
    for angle in [-180.0, -105.0, -75.0, -30.0, 30.0, 75.0, 105.0, 179.9]:
        matches = [
            sector for sector, ranges in DEFAULT_SECTOR_RANGES_DEG.items()
            if any(lo <= angle <= hi for lo, hi in ranges)
        ]
        assert matches, f"angle {angle} matched no sector"


# -- Front-sector safety banding (published only; nothing actuates) ---------------------


def test_safety_thresholds_classify_each_band():
    thresholds = SafetyThresholds(caution_distance_m=0.8, stop_distance_m=0.35,
                                  emergency_distance_m=0.15)

    assert thresholds.classify(2.0) is LidarSafetyLevel.NORMAL
    assert thresholds.classify(0.6) is LidarSafetyLevel.CAUTION
    assert thresholds.classify(0.30) is LidarSafetyLevel.STOP
    assert thresholds.classify(0.10) is LidarSafetyLevel.EMERGENCY


def test_missing_clearance_is_no_data_not_normal():
    # A blind sensor must never read as "path is clear".
    thresholds = SafetyThresholds()
    assert thresholds.classify(None) is LidarSafetyLevel.NO_DATA


def test_safety_thresholds_reject_inconsistent_ordering():
    try:
        SafetyThresholds(caution_distance_m=0.2, stop_distance_m=0.5, emergency_distance_m=0.1)
        assert False, "expected ValueError for stop > caution"
    except ValueError:
        pass


def test_analyzer_front_safety_level_uses_thresholds():
    analyzer = ObstacleSectorAnalyzer(
        obstacle_distance_m=0.8,
        min_points_for_valid_sector=1,
        safety_thresholds=SafetyThresholds(0.8, 0.35, 0.15),
    )

    sectors = analyzer.analyze(_packet([(0.0, 0.25)]))
    assert analyzer.get_front_safety_level(sectors) is LidarSafetyLevel.STOP

    sectors = analyzer.analyze(_packet([(0.0, 5.0)]))
    assert analyzer.get_front_safety_level(sectors) is LidarSafetyLevel.NORMAL


# -- MockLidar (simulation) -----------------------------------------------------


def test_mock_lidar_produces_scans_without_hardware():
    lidar = MockLidar(scan_frequency_hz=20.0, points_per_scan=180)
    lidar.initialize()
    lidar.start()
    try:
        packet = lidar.get_scan(timeout_s=2.0)

        assert packet is not None
        assert len(packet.points) == 180
        assert packet.sequence_id == 1
        assert packet.source == "mock_lidar"
        for p in packet.points:
            assert 0.12 <= p.distance_m <= 10.0
            assert -180.0 <= p.angle_deg < 180.0

        assert lidar.is_healthy() is True
    finally:
        lidar.shutdown()


def test_mock_lidar_status_reports_simulated():
    lidar = MockLidar(scan_frequency_hz=20.0)
    lidar.initialize()
    lidar.start()
    try:
        lidar.get_scan(timeout_s=2.0)
        status = lidar.get_status()

        assert status.simulated is True
        assert status.connected is True
        assert status.scanning is True
        assert status.healthy is True
        assert status.error_message is None
    finally:
        lidar.shutdown()


def test_mock_lidar_is_unhealthy_before_scanning_and_after_shutdown():
    lidar = MockLidar(scan_frequency_hz=20.0)
    lidar.initialize()
    assert lidar.is_healthy() is False       # initialized but not started

    lidar.start()
    lidar.get_scan(timeout_s=2.0)
    assert lidar.is_healthy() is True

    lidar.shutdown()
    assert lidar.is_healthy() is False
    assert lidar.get_status().connected is False


def test_mock_lidar_scan_contains_the_synthetic_obstacle():
    lidar = MockLidar(scan_frequency_hz=20.0, room_radius_m=4.0, obstacle_distance_m=0.6)
    lidar.initialize()
    lidar.start()
    try:
        packet = lidar.get_scan(timeout_s=2.0)
        assert packet is not None
        assert min(p.distance_m for p in packet.points) < 1.0   # the orbiting obstacle
        assert max(p.distance_m for p in packet.points) > 3.0   # the room wall
    finally:
        lidar.shutdown()
