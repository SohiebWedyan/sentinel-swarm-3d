"""Optional Semantic Reconstruction Layer (Phase 8).

ShapeR is never on the real-time navigation/perception critical path — see
base_shaper_client.py. Nothing in perception/, navigation/, or swarm/
imports from this package; the connection only ever goes the other way
(shaper reads finished tracks out of the SemanticMap).
"""
