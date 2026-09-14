"""Visualization and Monitoring Layer.

Modular by design: each panel (camera feed, depth map, detections/tracks,
future map/path/health panels) is a small function that takes plain data
(a frame, a depth array, a list of DetectedObject3D) and returns an image —
never a live object reference into the perception pipeline. That keeps
visualization strictly read-only and off the real-time safety path.
"""
