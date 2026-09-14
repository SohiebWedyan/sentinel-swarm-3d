"""Runtime orchestration: the multiprocessing process supervisor
(process_manager.py) and the cross-process "latest value" IPC primitive
(shared_state.py) that named processes (lidar_process.py today; camera_
/perception_/fusion_/localization_/mapping_/navigation_/robot_bridge_
process in later phases) are built on.

Nothing in perception/, geometry/, mapping/, navigation/, or hardware/
imports from this package — the dependency only goes this direction, so
every process target function can freely import from those layers.
"""
