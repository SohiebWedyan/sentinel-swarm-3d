"""Perception Layer.

RGB frame -> {object detections, depth map} -> 3D fusion -> tracking.
Every stage is behind an interface (detector/base_detector.py,
depth/base_depth_estimator.py, tracking/base_tracker.py) so the concrete
model backing each one (YOLO version, DA3 vs stereo vs RGB-D, ByteTrack vs
BoT-SORT) can be swapped from config alone.
"""
