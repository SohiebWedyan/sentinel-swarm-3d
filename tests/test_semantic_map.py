"""Unit tests for the SemanticMap's confidence-weighted fusion, since the
spec is explicit that map updates must never blindly overwrite existing
objects.
"""

from __future__ import annotations

from common.types import BBox, DetectedObject3D
from mapping.semantic_map import SemanticMap


def _obj(x: float, confidence: float, class_name: str = "person") -> DetectedObject3D:
    return DetectedObject3D(
        object_id=f"obj-{x}-{confidence}",
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        camera_coords=(0.0, 0.0, 1.0),
        world_coords=(x, 0.0, 0.0),
        bbox=BBox(0, 0, 10, 10),
    )


def test_add_observation_creates_new_object():
    smap = SemanticMap(merge_distance_m=0.5)
    smap.add_observation(_obj(x=1.0, confidence=0.8))

    exported = smap.export_map()
    assert len(exported) == 1
    assert exported[0]["class_name"] == "person"


def test_nearby_same_class_observation_is_fused_not_duplicated():
    smap = SemanticMap(merge_distance_m=0.5)
    smap.add_observation(_obj(x=1.0, confidence=0.8))
    smap.add_observation(_obj(x=1.1, confidence=0.6))  # within merge_distance_m

    exported = smap.export_map()
    assert len(exported) == 1
    assert exported[0]["observations"] == 2


def test_far_observation_creates_a_second_object():
    smap = SemanticMap(merge_distance_m=0.5)
    smap.add_observation(_obj(x=1.0, confidence=0.8))
    smap.add_observation(_obj(x=5.0, confidence=0.8))  # far away

    assert len(smap.export_map()) == 2


def test_remove_stale_objects_drops_old_entries():
    smap = SemanticMap(merge_distance_m=0.5, stale_after_s=10.0)
    obj = _obj(x=1.0, confidence=0.8)
    obj.timestamp = 0.0
    object_id = smap.add_observation(obj)
    smap._objects[object_id].last_seen = 0.0

    removed = smap.remove_stale_objects(now=100.0)

    assert removed == 1
    assert smap.export_map() == []
