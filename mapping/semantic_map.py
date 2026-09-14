"""Semantic object map.

Implements the SemanticMap interface from the spec. The occupancy-map and
raw-point-cloud sides of the "hybrid mapping system" are Phase 4 work
(they depend on the navigation layer's grid representation) and are not
built here; this module only owns the semantic object database, which
Phase 1/2 can already exercise standalone (e.g. a script that runs the
Phase 1 perception loop and prints export_map() every N seconds).

Fusion across robots (Phase 7) will call add_observation() once per
incoming swarm/*/objects MQTT message, using the same confidence-based
merge logic single-robot re-observations already use here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from common.types import DetectedObject3D

logger = logging.getLogger(__name__)


@dataclass
class SemanticObjectRecord:
    object_id: str
    class_name: str
    confidence: float
    position_world: tuple[float, float, float]
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dimensions: tuple[float, float, float] | None = None
    velocity: tuple[float, float, float] | None = None
    last_seen: float = field(default_factory=time.time)
    source_robot: str = "unknown"
    observations: int = 1

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "position_world": list(self.position_world),
            "orientation": list(self.orientation),
            "dimensions": list(self.dimensions) if self.dimensions else None,
            "velocity": list(self.velocity) if self.velocity else None,
            "last_seen": self.last_seen,
            "source_robot": self.source_robot,
            "observations": self.observations,
        }


class SemanticMap:
    def __init__(self, merge_distance_m: float = 0.5, stale_after_s: float = 30.0) -> None:
        """merge_distance_m: two observations of the same class within this
        world-frame distance are treated as the same object rather than
        added as a new one (a coarse stand-in for full track-to-map data
        association, sufficient for a single-robot Phase 1/2 map).
        """
        self.merge_distance_m = merge_distance_m
        self.stale_after_s = stale_after_s
        self._objects: dict[str, SemanticObjectRecord] = {}

    def add_observation(self, obj: DetectedObject3D) -> str:
        """Add or fuse one fresh observation. Returns the map object_id it
        ended up under (may differ from obj.object_id if it was merged into
        an existing record).
        """
        if obj.world_coords is None:
            logger.debug("Dropping observation of %s with no world_coords", obj.class_name)
            return ""

        existing = self._find_nearby_same_class(obj.class_name, obj.world_coords)
        if existing is None:
            record = SemanticObjectRecord(
                object_id=obj.object_id,
                class_name=obj.class_name,
                confidence=obj.confidence,
                position_world=obj.world_coords,
                dimensions=obj.dimensions,
                velocity=obj.velocity,
                source_robot=obj.source_robot,
            )
            self._objects[record.object_id] = record
            return record.object_id

        self.update_object(existing.object_id, obj)
        return existing.object_id

    def update_object(self, object_id: str, obj: DetectedObject3D) -> None:
        """Confidence-weighted fusion of a new observation into an existing
        record — never a blind overwrite, per the spec.
        """
        record = self._objects.get(object_id)
        if record is None or obj.world_coords is None:
            return

        total_weight = record.confidence + obj.confidence
        if total_weight <= 0:
            return

        w_old, w_new = record.confidence / total_weight, obj.confidence / total_weight
        fused_position = tuple(
            w_old * o + w_new * n for o, n in zip(record.position_world, obj.world_coords)
        )

        record.position_world = fused_position
        record.confidence = max(record.confidence, obj.confidence)
        record.last_seen = obj.timestamp
        record.observations += 1
        if obj.dimensions:
            record.dimensions = obj.dimensions
        if obj.velocity:
            record.velocity = obj.velocity

    def merge_objects(self, object_id_a: str, object_id_b: str) -> None:
        """Merge two map entries later discovered to be the same physical
        object (e.g. after multi-robot fusion resolves an ambiguity).
        Keeps the higher-confidence record's identity.
        """
        a, b = self._objects.get(object_id_a), self._objects.get(object_id_b)
        if a is None or b is None:
            return
        keep, drop = (a, b) if a.confidence >= b.confidence else (b, a)
        keep.observations += drop.observations
        keep.confidence = max(keep.confidence, drop.confidence)
        keep.last_seen = max(keep.last_seen, drop.last_seen)
        del self._objects[drop.object_id]

    def remove_stale_objects(self, now: float | None = None) -> int:
        """Drop objects not seen in stale_after_s. Returns the count removed."""
        now = now if now is not None else time.time()
        stale_ids = [oid for oid, r in self._objects.items() if now - r.last_seen > self.stale_after_s]
        for oid in stale_ids:
            del self._objects[oid]
        return len(stale_ids)

    def get_nearby_objects(self, position_world: tuple[float, float, float], radius_m: float) -> list[SemanticObjectRecord]:
        result = []
        for record in self._objects.values():
            dist = sum((a - b) ** 2 for a, b in zip(record.position_world, position_world)) ** 0.5
            if dist <= radius_m:
                result.append(record)
        return result

    def export_map(self) -> list[dict]:
        return [r.to_dict() for r in self._objects.values()]

    def _find_nearby_same_class(self, class_name: str, position: tuple[float, float, float]):
        for record in self._objects.values():
            if record.class_name != class_name:
                continue
            dist = sum((a - b) ** 2 for a, b in zip(record.position_world, position)) ** 0.5
            if dist <= self.merge_distance_m:
                return record
        return None
