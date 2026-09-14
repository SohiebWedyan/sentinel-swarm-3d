"""Mapping Layer (Phase 3+ for the semantic side, Phase 4 for occupancy).

SemanticMap below is a working, minimal in-memory implementation of the
object side of the "hybrid mapping system" (occupancy map + point cloud +
semantic object database) — it's useful standalone for Phase 1/2 debugging
(e.g. dumping what a single robot has fused/tracked so far) even before
multi-robot fusion (Phase 7) or the occupancy grid (Phase 4) exist.
"""
