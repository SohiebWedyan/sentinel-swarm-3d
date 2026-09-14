"""Navigation Layer (Phase 4+).

Interfaces only in Phase 1 (system.yaml layers.navigation: false). The
hierarchical Goal -> Global Planner -> Local Planner -> Velocity Command ->
Motor Controller pipeline from the spec is defined here so the perception
side (already running in Phase 1) has a stable contract to feed once
Phase 4 turns navigation on.
"""
