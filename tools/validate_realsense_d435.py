#!/usr/bin/env python3
"""Run the D435-only validation and write a JSON evidence report.

No motor, navigation, SLAM, LiDAR, IMU or swarm code is started by this tool.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config_loader import load_system_config
from common.logging_utils import configure_logging
from hardware.realsense_d435 import RealSenseD435Validator


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Intel RealSense D435 RGB-D hardware")
    parser.add_argument("--config", default="config/system.yaml")
    parser.add_argument("--out", default="logs", help="Directory for the JSON report")
    args = parser.parse_args()
    cfg = load_system_config(args.config)
    configure_logging(cfg.get("system", {}).get("log_level", "INFO"), "leader", args.out)
    d435_cfg = cfg.get("camera", {}).get("realsense_d435", {})
    report = RealSenseD435Validator(d435_cfg).run()
    payload = report.to_dict() | {"generated_at": datetime.now(timezone.utc).isoformat()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"d435_validation_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for check in report.checks:
        print(f"{check.name:22} {check.status:4} {check.detail}")
    print(f"Overall: {report.overall_status}; report: {report_path}")
    return 0 if report.overall_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
