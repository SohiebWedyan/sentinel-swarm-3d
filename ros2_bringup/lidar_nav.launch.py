#!/usr/bin/env python3
"""
SentinelSwarm-3D — LiDAR SLAM + Nav2, one command.

    ros2 launch ./lidar_nav.launch.py

Startup order (delays are load-bearing — getting this wrong leaves the TF
tree split into two unconnected trees, which shows up as a missing "map"):

    0s   RPLIDAR C1 driver            -> /scan at 10 Hz
    0s   static TF base_link->laser
    3s   ICP odometry                 -> /odom, odom->base_link
    6s   RTAB-Map SLAM                -> map->odom, /map
    9s   rtabmap_viz
    12s  Nav2                         -> costmaps, planner, controller

Overrides:
    nav2:=false          SLAM only, skip Nav2
    viz:=false           no GUI (lighter)
    rviz:=true           RViz instead of rtabmap_viz — needed to click goals
    lidar_z:=0.06        LiDAR height above base_link, in metres
    port:=/dev/ttyUSB1
    params_file:=/abs/path/nav2_lidar_params.yaml

Nothing moves: there is no motor driver yet. Nav2 publishes velocity
commands on /cmd_vel_nav, which is what you watch to confirm planning works.

LiDAR-only: no camera, no encoders, no IMU. The camera layer is added later
as a second costmap observation source.
"""

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    port = LaunchConfiguration("port")
    baud = LaunchConfiguration("baud")
    lidar_z = LaunchConfiguration("lidar_z")
    viz = LaunchConfiguration("viz")
    rviz = LaunchConfiguration("rviz")
    nav2 = LaunchConfiguration("nav2")
    params_file = LaunchConfiguration("params_file")
    delete_db = LaunchConfiguration("delete_db")

    args = [
        DeclareLaunchArgument("port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("baud", default_value="460800",
                              description="C1 uses 460800"),
        DeclareLaunchArgument("lidar_z", default_value="0.15",
                              description="measure on the real chassis"),
        DeclareLaunchArgument("viz", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("nav2", default_value="true"),
        DeclareLaunchArgument("delete_db", default_value="true"),
        DeclareLaunchArgument("params_file",
                              default_value="nav2_lidar_params.yaml"),
    ]

    # ICP registration, constrained to the plane. A 2D LiDAR cannot observe
    # z, roll or pitch, so leaving those free makes the estimate drift.
    common = {
        "frame_id": "base_link",
        "subscribe_depth": False,
        "subscribe_rgb": False,
        "subscribe_scan": True,
        "approx_sync": True,
        "use_sim_time": False,
        "Reg/Strategy": "1",
        "Reg/Force3DoF": "true",
        "Icp/VoxelSize": "0.05",
        "Icp/MaxCorrespondenceDistance": "0.1",
        "Icp/PointToPlane": "false",
        "Icp/Iterations": "10",
        "Icp/Epsilon": "0.001",
        "Icp/MaxTranslation": "2.0",
        "Icp/CorrespondenceRatio": "0.3",
        "Grid/FromDepth": "false",
        "Grid/Sensor": "0",
        "Grid/RangeMax": "12.0",
        "Grid/CellSize": "0.05",
    }

    lidar = Node(
        package="rplidar_ros",
        executable="rplidar_node",
        name="rplidar_node",
        output="screen",
        parameters=[{
            "serial_port": port,
            "serial_baudrate": baud,
            "frame_id": "laser",
            "scan_mode": "Standard",
            "angle_compensate": True,
            "inverted": False,
        }],
    )

    base_to_laser = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_laser",
        output="log",
        arguments=["--x", "0", "--y", "0", "--z", lidar_z,
                   "--roll", "0", "--pitch", "0", "--yaw", "0",
                   "--frame-id", "base_link",
                   "--child-frame-id", "laser"],
    )

    # scan_cloud is remapped to an unused name on purpose: icp_odometry
    # subscribes to both scan and scan_cloud and waits for whichever is
    # published. Leaving scan_cloud live but empty hangs the node forever.
    icp_odom = Node(
        package="rtabmap_odom",
        executable="icp_odometry",
        name="icp_odometry",
        output="screen",
        parameters=[{
            "frame_id": "base_link",
            "odom_frame_id": "odom",
            "publish_tf": True,
            "wait_for_transform": 0.2,
            "approx_sync": True,
            "Reg/Strategy": "1",
            "Reg/Force3DoF": "true",
            "Icp/VoxelSize": "0.05",
            "Icp/MaxCorrespondenceDistance": "0.1",
            "Icp/PointToPlane": "false",
            "Icp/Iterations": "10",
            "Icp/Epsilon": "0.001",
            "Icp/CorrespondenceRatio": "0.2",
            "Odom/Strategy": "0",
            "Odom/GuessMotion": "true",
            "Odom/ResetCountdown": "1",
            "OdomF2M/ScanSubtractRadius": "0.05",
            "OdomF2M/ScanMaxSize": "15000",
        }],
        remappings=[("scan", "/scan"),
                    ("scan_cloud", "/scan_cloud_unused")],
    )

    # Every RTAB-Map internal parameter is a STRING, including the booleans.
    # Passing Python True/False raises InvalidParameterTypeException and the
    # node dies on startup.
    rtabmap = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        parameters=[common, {
            "Mem/IncrementalMemory": "true",
            "Mem/InitWMWithAllNodes": "false",
            "RGBD/NeighborLinkRefining": "true",
            "RGBD/ProximityBySpace": "true",
            "RGBD/ProximityPathMaxNeighbors": "1",
            "RGBD/AngularUpdate": "0.05",
            "RGBD/LinearUpdate": "0.05",
            "RGBD/OptimizeFromGraphEnd": "false",
            "Optimizer/Slam2D": "true",
        }],
        # Nav2's static layer reads /map, so publish the grid there.
        remappings=[("scan", "/scan"), ("odom", "/odom"),
                    ("grid_map", "/map")],
        arguments=[PythonExpression([
            "'--delete_db_on_start' if '", delete_db, "' == 'true' else ''"])],
    )

    rtabmap_viz = Node(
        condition=IfCondition(PythonExpression([
            "'", viz, "' == 'true' and '", rviz, "' == 'false'"])),
        package="rtabmap_viz",
        executable="rtabmap_viz",
        name="rtabmap_viz",
        output="log",
        parameters=[common],
        remappings=[("scan", "/scan"), ("odom", "/odom")],
    )

    # RViz is the one that lets you click "2D Goal Pose".
    rviz_node = Node(
        condition=IfCondition(rviz),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", PathJoinSubstitution([
            FindPackageShare("nav2_bringup"), "rviz",
            "nav2_default_view.rviz"])],
    )

    # No AMCL, no map_server: RTAB-Map already owns map->odom and /map.
    nav2_stack = GroupAction(
        condition=IfCondition(nav2),
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare("nav2_bringup"), "launch",
                "navigation_launch.py"])),
            launch_arguments={
                "use_sim_time": "false",
                "params_file": params_file,
                "autostart": "true",
            }.items(),
        )],
    )

    return LaunchDescription(args + [
        lidar,
        base_to_laser,
        TimerAction(period=3.0, actions=[icp_odom]),
        TimerAction(period=6.0, actions=[rtabmap]),
        TimerAction(period=9.0, actions=[rtabmap_viz, rviz_node]),
        # Nav2 last: its costmaps need map->odom->base_link to already exist.
        TimerAction(period=12.0, actions=[nav2_stack]),
    ])
