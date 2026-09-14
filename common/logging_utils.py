"""Structured logging setup shared by every entry point.

Rationale: the real-time safety rules in the spec require that failures
(camera drop, inference exception, MQTT disconnect) are visible and
attributable to a layer/robot, not swallowed. A single configure_logging()
call gives every module's `logging.getLogger(__name__)` a consistent,
greppable format.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


class _ContextFilter(logging.Filter):
    """Injects a robot_id into every record so multi-robot logs are attributable."""

    def __init__(self, robot_id: str) -> None:
        super().__init__()
        self.robot_id = robot_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.robot_id = self.robot_id
        return True


def configure_logging(
    level: str = "INFO",
    robot_id: str = "unknown",
    log_dir: str | None = None,
) -> None:
    """Idempotent-ish global logging configuration.

    Call once from main.py (or a test's setup) before any other module logs.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if configure_logging() is called more than once
    # (e.g. re-entrant tests).
    root.handlers.clear()

    fmt = "%(asctime)s | %(levelname)-8s | %(robot_id)s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)
    context_filter = _ContextFilter(robot_id)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(context_filter)
    root.addHandler(stream_handler)

    if log_dir:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(Path(log_dir) / f"{robot_id}.log")
            file_handler.setFormatter(formatter)
            file_handler.addFilter(context_filter)
            root.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - filesystem edge case
            root.warning("Could not create log file in %s: %s", log_dir, exc)
