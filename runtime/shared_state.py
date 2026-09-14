"""Multiprocessing-safe "latest value" primitive.

Per the spec's shared-data rule: never let a queue between processes grow
unbounded — always prefer latest-value semantics, dropping old values a
slow consumer hasn't kept up with. This is the same drop-oldest pattern
perception/depth/da3_depth_estimator.py already uses with an in-process
queue.Queue, generalized to work across a real OS process boundary via
multiprocessing.Queue.

Single-writer, single-reader use (one process publishes, one/more
processes/threads read the latest value) needs no additional locking here
— get_nowait()+put_nowait() on a maxsize=1 Queue is what keeps it bounded.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import time
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class LatestValueQueue(Generic[T]):
    def __init__(self) -> None:
        self._queue: "mp.Queue[T]" = mp.Queue(maxsize=1)

    def put_latest(self, value: T) -> None:
        """Publish a new value, discarding whatever hadn't been consumed yet."""
        try:
            self._queue.get_nowait()
        except _queue.Empty:
            pass
        try:
            self._queue.put_nowait(value)
        except _queue.Full:
            pass  # extremely unlikely race with a concurrent reader; drop this update

    def get_latest(self) -> Optional[T]:
        """Non-blocking read. Returns None if nothing has been published yet."""
        try:
            return self._queue.get_nowait()
        except _queue.Empty:
            return None

    def put_latest_sync(self, value: T, timeout: float = 1.0) -> bool:
        """Publish and wait until the value has actually reached the pipe.

        multiprocessing.Queue hands writes to a background feeder thread, so
        put_latest() returns before the data is really transferred. A
        publisher that puts one final message and then exits can therefore
        lose it entirely — which is exactly the case that matters most: the
        status explaining WHY a process is bailing out.

        Continuous publishers don't need this (the next tick republishes);
        use it only for terminal/one-shot messages. Returns False if the
        value could not be confirmed within `timeout`.
        """
        self.put_latest(value)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._queue.empty():
                return True
            time.sleep(0.01)
        return False

    def get_latest_blocking(self, timeout: float = 1.0) -> Optional[T]:
        """Blocking read with a timeout. Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except _queue.Empty:
            return None
