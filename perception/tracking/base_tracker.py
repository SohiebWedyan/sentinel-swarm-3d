"""Object tracker abstraction.

Sits after fusion in the pipeline: takes one frame's DetectedObject3D list
(freshly created, each with a brand-new object_id) and returns them
annotated with a persistent track_id and tracking_state, merging repeat
observations of the same physical object instead of letting the semantic
map fill up with duplicates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from common.types import DetectedObject3D


class BaseTracker(ABC):
    """Any multi-object tracker (ByteTrack, BoT-SORT, or a simple fallback)."""

    @abstractmethod
    def update(self, objects: list[DetectedObject3D]) -> list[DetectedObject3D]:
        """Associate this frame's fused objects with existing tracks.

        Returns the same objects with track_id and tracking_state set
        (NEW on first observation, TRACKED on a matched re-observation).
        Tracks not matched this frame age out internally and eventually
        transition to LOST/REMOVED — see get_active_tracks().
        """

    @abstractmethod
    def get_active_tracks(self) -> list[DetectedObject3D]:
        """Return the tracker's current best estimate for every track still
        considered active (TRACKED or recently LOST, within the buffer
        window), including ones not re-observed in the latest frame.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all track state (e.g. on a scene change)."""
