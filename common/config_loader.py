"""YAML configuration loading.

Every module in the project takes its configuration as a plain dict (or a
small typed wrapper) rather than reaching into a global singleton — this is
what config_loader hands back. That keeps modules testable in isolation
(tests just pass a hand-built dict) while main.py wires the real files.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file relative to the project root (or absolute)."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file did not parse to a mapping: {p}")

    return data


def load_system_config(system_path: str | Path = "config/system.yaml") -> dict[str, Any]:
    """Load system.yaml and merge in every file it references under
    config_files, keyed by the same top-level key each file uses
    (camera.yaml -> {"camera": {...}}, etc).

    Returns a single merged dict, e.g.:
        {
          "system": {...},
          "camera": {...},
          "perception": {...},
          "navigation": {...},
          "mqtt": {...},
          "robots": {...},
        }
    """
    system_cfg = load_yaml(system_path)
    merged: dict[str, Any] = copy.deepcopy(system_cfg)

    config_files = system_cfg.get("config_files", {})
    for key, rel_path in config_files.items():
        try:
            sub_cfg = load_yaml(rel_path)
        except FileNotFoundError:
            logger.warning("Referenced config '%s' (%s) not found; skipping.", key, rel_path)
            continue
        merged.update(sub_cfg)

    return merged


def get(cfg: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    """Convenience accessor: get(cfg, "perception.detector.device")."""
    node: Any = cfg
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
