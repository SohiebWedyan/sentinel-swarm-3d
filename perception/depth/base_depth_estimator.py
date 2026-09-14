"""Depth estimation abstraction.

This is the interface referenced throughout the spec: the navigation and
fusion layers depend only on this, never on Depth Anything 3 directly, so
DA3 can later be swapped for stereo depth, an RGB-D camera, or a future
model without touching anything downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from common.types import DepthKind


class BaseDepthEstimator(ABC):
    """Any monocular/stereo/RGB-D depth source.

    Implementations that run heavy inference (like DA3 on Jetson) are
    expected to do so asynchronously internally — infer() may just enqueue
    a frame, and get_depth() returns the most recent completed result
    rather than blocking. See DA3DepthEstimator for the reference pattern.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Load model weights / allocate the inference backend (GPU context, etc)."""

    @abstractmethod
    def infer(self, frame: np.ndarray) -> None:
        """Submit a frame for depth inference. Must not block the caller for
        longer than a bounded, small amount of time — see async_processing
        in config/perception.yaml. Implementations that can't keep up should
        drop frames rather than build an unbounded queue.
        """

    @abstractmethod
    def get_depth(self) -> Optional[np.ndarray]:
        """Return the most recent available depth map (meters, HxW float32),
        or None if no result is available yet.
        """

    @abstractmethod
    def get_confidence(self) -> Optional[np.ndarray]:
        """Return a per-pixel confidence/uncertainty map aligned with
        get_depth()'s output, or None if the backend doesn't provide one.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release model/GPU resources and stop any background thread."""

    # -- optional, concrete extensions -----------------------------------------------------

    def infer_sync(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run depth on THIS frame and return THIS frame's result.

        The async infer()/get_depth() pair above is right for a control loop
        that must never block. It is wrong when a caller needs the depth map
        that corresponds to a specific frame — the perception process pairs
        depth with the detections from the same frame, and a stale map would
        silently mis-associate the two.

        The default implementation is best-effort (submit, then read
        whatever is latest) and may return a stale or None result;
        implementations that can run synchronously should override it.
        """
        self.infer(frame)
        return self.get_depth()

    def get_depth_kind(self) -> DepthKind:
        """What this estimator's numbers MEAN — see common.types.DepthKind.

        Defaults to UNKNOWN deliberately: an estimator that has not declared
        its units must never be assumed to produce meters.
        """
        return DepthKind.UNKNOWN

    def get_model_info(self) -> dict:
        """Provenance for diagnostics (model, device, precision, backend)."""
        return {}

    def __enter__(self) -> "BaseDepthEstimator":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
