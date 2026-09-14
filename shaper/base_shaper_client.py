"""ShapeR client interface — optional, asynchronous, background-only.

Design constraint from the spec, restated as code structure: this interface
has no method that the real-time navigation loop could accidentally call
synchronously. submit_object() is fire-and-forget; results arrive later via
poll_results(), consumed by whatever background task feeds them into the
SemanticMap (kept separate from the real-time occupancy/track representation).

Gating logic (enough views, known camera poses, high confidence, object
stability) belongs in the caller that decides *when* to call submit_object
— e.g. a background "reconstruction candidate" watcher over the
SemanticMap — not in this client itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ShapeRSubmission:
    object_id: str
    class_name: str
    views: list  # list of image crops / frame references
    camera_poses: list  # Pose3D per view, same order as `views`
    sparse_geometry: list | None = None  # optional sparse point cloud seed


@dataclass
class ShapeRResult:
    object_id: str
    mesh_path: str
    success: bool
    error: str | None = None


class BaseShaperClient(ABC):
    @abstractmethod
    def submit_object(self, submission: ShapeRSubmission) -> str:
        """Enqueue a reconstruction job. Returns a job id immediately;
        must never block on network I/O or the reconstruction itself.
        """

    @abstractmethod
    def poll_results(self) -> list[ShapeRResult]:
        """Return any reconstruction results completed since the last poll.
        Called from a background task, never from the navigation loop.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Whether a ShapeR server is currently reachable. Navigation and
        perception must behave identically whether this returns True or
        False — it only gates whether reconstruction jobs get submitted.
        """
