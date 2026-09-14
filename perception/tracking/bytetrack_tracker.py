"""ByteTrack-based tracker, with a dependency-free IOU tracker fallback.

Uses the `supervision` package's ByteTrack implementation when available
(config perception.tracker.backend == "bytetrack"/"botsort" maps to it);
otherwise falls back to a small hand-rolled IOU + confidence-weighted
tracker so the pipeline still gets persistent IDs, appear/disappear/
reappear handling, and duplicate suppression with zero extra dependencies.

Both paths implement BaseTracker, so main.py never branches on which one is
active.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from common.types import BBox, DetectedObject3D, TrackingState
from perception.tracking.base_tracker import BaseTracker

logger = logging.getLogger(__name__)

try:
    import supervision as sv

    _HAS_SUPERVISION = True
except ImportError:  # pragma: no cover
    _HAS_SUPERVISION = False
    logger.info("supervision not installed; using the built-in IOU tracker fallback.")


def _iou(a: BBox, b: BBox) -> float:
    xx1, yy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    xx2, yy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter_w, inter_h = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = inter_w * inter_h
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    last_object: DetectedObject3D
    hits: int = 1
    misses: int = 0
    state: TrackingState = TrackingState.NEW
    last_seen: float = field(default_factory=time.time)
    world_position_history: list[tuple[float, float, float]] = field(default_factory=list)


class _SimpleIOUTracker:
    """Greedy IOU matching + a small confirm/miss state machine.

    Deliberately simple (no Kalman filter) so it has no numeric edge cases
    to debug in Phase 1; kalman_filter=True in config is honored by
    ByteTrack when `supervision` is installed, and this fallback documents
    where a real Kalman filter would slot in (see _predict()).
    """

    def __init__(self, match_thresh: float, track_buffer_frames: int, min_hits_to_confirm: int):
        self.match_thresh = match_thresh
        self.track_buffer_frames = track_buffer_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def _predict(self, track: _Track) -> BBox:
        """Placeholder for a Kalman motion prediction step. Currently just
        returns the last known box (zero-velocity model), which is enough
        for the IOU gate at the frame rates this runs at.
        """
        return track.last_object.bbox

    def update(self, objects: list[DetectedObject3D]) -> list[DetectedObject3D]:
        unmatched_objects = list(range(len(objects)))
        matched_track_ids: set[int] = set()

        for track_id, track in self._tracks.items():
            if not objects:
                break
            predicted_bbox = self._predict(track)
            best_idx, best_iou = -1, 0.0
            for idx in unmatched_objects:
                if objects[idx].bbox is None:
                    continue
                score = _iou(predicted_bbox, objects[idx].bbox)
                if score > best_iou:
                    best_idx, best_iou = idx, score

            if best_idx >= 0 and best_iou >= self.match_thresh:
                obj = objects[best_idx]
                track.last_object = obj
                track.hits += 1
                track.misses = 0
                track.last_seen = time.time()
                if obj.world_coords:
                    track.world_position_history.append(obj.world_coords)
                track.state = (
                    TrackingState.TRACKED if track.hits >= self.min_hits_to_confirm else TrackingState.NEW
                )
                obj.track_id = track_id
                obj.tracking_state = track.state
                obj.observations = track.hits
                matched_track_ids.add(track_id)
                unmatched_objects.remove(best_idx)

        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids:
                track.misses += 1
                track.state = TrackingState.LOST if track.misses <= self.track_buffer_frames else TrackingState.REMOVED

        self._tracks = {
            tid: t for tid, t in self._tracks.items() if t.state != TrackingState.REMOVED
        }

        # Duplicate suppression: two *new* objects this frame that overlap
        # heavily are almost certainly the same physical object seen twice
        # (e.g. overlapping class predictions) — keep the higher-confidence one.
        remaining = [objects[i] for i in unmatched_objects]
        deduped = self._suppress_duplicates(remaining)

        for obj in deduped:
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _Track(track_id=track_id, last_object=obj)
            obj.track_id = track_id
            obj.tracking_state = TrackingState.NEW
            obj.observations = 1

        return objects

    def _suppress_duplicates(self, objects: list[DetectedObject3D]) -> list[DetectedObject3D]:
        kept: list[DetectedObject3D] = []
        for obj in sorted(objects, key=lambda o: o.confidence, reverse=True):
            if obj.bbox is None:
                kept.append(obj)
                continue
            if any(o.bbox is not None and _iou(obj.bbox, o.bbox) > 0.7 for o in kept):
                continue
            kept.append(obj)
        return kept

    def get_active_tracks(self) -> list[DetectedObject3D]:
        return [t.last_object for t in self._tracks.values() if t.state != TrackingState.REMOVED]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


class ByteTrackTracker(BaseTracker):
    """Wraps `supervision.ByteTrack` when available, else _SimpleIOUTracker."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer_frames: int = 30,
        min_hits_to_confirm: int = 3,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer_frames = track_buffer_frames
        self.min_hits_to_confirm = min_hits_to_confirm

        self._fallback = _SimpleIOUTracker(match_thresh, track_buffer_frames, min_hits_to_confirm)
        self._sv_tracker = None
        if _HAS_SUPERVISION:
            try:
                self._sv_tracker = sv.ByteTrack(
                    track_activation_threshold=track_thresh,
                    lost_track_buffer=track_buffer_frames,
                    minimum_matching_threshold=match_thresh,
                    minimum_consecutive_frames=min_hits_to_confirm,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to construct supervision.ByteTrack (%s); "
                               "using IOU fallback instead.", exc)
                self._sv_tracker = None

    def update(self, objects: list[DetectedObject3D]) -> list[DetectedObject3D]:
        if self._sv_tracker is None or not objects:
            return self._fallback.update(objects)

        boxes = np.array([o.bbox.as_xyxy() for o in objects if o.bbox is not None], dtype=np.float32)
        confs = np.array([o.confidence for o in objects if o.bbox is not None], dtype=np.float32)
        class_ids = np.array([o.class_id for o in objects if o.bbox is not None], dtype=int)

        if boxes.shape[0] == 0:
            return objects

        detections = sv.Detections(xyxy=boxes, confidence=confs, class_id=class_ids)
        try:
            tracked = self._sv_tracker.update_with_detections(detections)
        except Exception as exc:  # noqa: BLE001
            logger.warning("supervision ByteTrack update failed (%s); falling back to IOU "
                           "tracker for this frame.", exc)
            return self._fallback.update(objects)

        for obj, track_id in zip(objects, tracked.tracker_id):
            obj.track_id = int(track_id) if track_id is not None else None
            obj.tracking_state = TrackingState.TRACKED if track_id is not None else TrackingState.NEW

        return objects

    def get_active_tracks(self) -> list[DetectedObject3D]:
        if self._sv_tracker is None:
            return self._fallback.get_active_tracks()
        # supervision's ByteTrack doesn't expose a "last known state per
        # track" query the way our fallback does; Phase 1's visualization
        # only needs the per-frame update() output, so this is a light stub
        # that later phases (SemanticMap) can extend if needed.
        return []

    def reset(self) -> None:
        self._fallback.reset()
        if _HAS_SUPERVISION:
            try:
                self._sv_tracker = sv.ByteTrack(
                    track_activation_threshold=self.track_thresh,
                    lost_track_buffer=self.track_buffer_frames,
                    minimum_matching_threshold=self.match_thresh,
                    minimum_consecutive_frames=self.min_hits_to_confirm,
                )
            except Exception:  # noqa: BLE001
                self._sv_tracker = None


def build_tracker_from_config(cfg: dict) -> BaseTracker:
    t_cfg = cfg["perception"]["tracker"]
    return ByteTrackTracker(
        track_thresh=t_cfg.get("track_thresh", 0.5),
        match_thresh=t_cfg.get("match_thresh", 0.8),
        track_buffer_frames=t_cfg.get("track_buffer_frames", 30),
        min_hits_to_confirm=t_cfg.get("min_hits_to_confirm", 3),
    )
