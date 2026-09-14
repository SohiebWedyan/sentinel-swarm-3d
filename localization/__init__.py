"""Localization Layer (Phase 3+).

Only the interface is implemented in Phase 1 — main.py does not instantiate
a PoseEstimator yet (system.yaml layers.localization: false). It exists now
so the geometry layer's camera_to_world() has a stable contract to target,
and so no downstream code has to change shape when Phase 3 lands.
"""
