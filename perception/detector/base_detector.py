"""Object detector abstraction.

Mirrors the depth estimator interface style (initialize/infer/shutdown) so
both halves of the perception fork in the spec's diagram — "YOLO Object
Detection" and "Depth Anything 3" — are swappable the same way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from common.types import Detection2D


class BaseObjectDetector(ABC):
    """Any 2D object detector (YOLO today; anything else tomorrow)."""

    @abstractmethod
    def initialize(self) -> None:
        """Load model weights / allocate the inference backend."""

    @abstractmethod
    def infer(self, frame: np.ndarray) -> list[Detection2D]:
        """Run detection on a single BGR frame, return raw detections."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release model/GPU resources."""

    def __enter__(self) -> "BaseObjectDetector":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
