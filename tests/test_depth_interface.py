"""Contract test for BaseDepthEstimator implementations, run against
DA3DepthEstimator's mock backend (no GPU/model weights required — this is
exactly the "no real DA3 available" path every non-Jetson dev machine and
CI runner will take).
"""

from __future__ import annotations

import time

import numpy as np

from perception.depth.da3_depth_estimator import DA3DepthEstimator


def test_da3_estimator_produces_depth_after_a_frame_via_mock_backend():
    estimator = DA3DepthEstimator(
        model_size="small",
        device="cpu",
        inference_resolution=(64, 48),
        inference_frequency_hz=20.0,
        max_queue_size=2,
    )
    estimator.initialize()
    try:
        assert estimator.is_mock is True  # no real DA3 backend on this machine

        frame = np.random.randint(0, 255, size=(240, 320, 3), dtype=np.uint8)
        estimator.infer(frame)

        depth = None
        for _ in range(50):  # poll for the async worker to produce a result
            depth = estimator.get_depth()
            if depth is not None:
                break
            time.sleep(0.05)

        assert depth is not None
        assert depth.shape == (48, 64)
        assert np.isfinite(depth).all()
        assert depth.min() >= 0.0
    finally:
        estimator.shutdown()


def test_da3_estimator_get_depth_is_none_before_any_inference():
    estimator = DA3DepthEstimator(inference_resolution=(32, 24))
    estimator.initialize()
    try:
        assert estimator.get_depth() is None
    finally:
        estimator.shutdown()


def test_da3_estimator_infer_is_non_blocking_under_queue_pressure():
    estimator = DA3DepthEstimator(inference_resolution=(32, 24), max_queue_size=1)
    estimator.initialize()
    try:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        start = time.time()
        for _ in range(20):
            estimator.infer(frame)  # must never block even if the worker is slower
        elapsed = time.time() - start
        assert elapsed < 1.0
    finally:
        estimator.shutdown()
