#!/usr/bin/env python3
"""Aggregate Micro-F1 across seeds and print mean and sample standard deviation."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np


def _load(path: Path, group: str) -> float:
    payload = json.loads(path.read_text())
    if group == "overall":
        section = payload.get("test") or payload.get("overall") or {}
        if "overall" in section:
            section = section["overall"]
        return float(section["micro_f1"] if "micro_f1" in section else section["f1"])
    section = payload.get("test", {}).get(group) or payload.get("groups", {}).get(group)
    if section is None:
        raise KeyError(f"{path} has no group {group}")
    return float(section["micro_f1"] if "micro_f1" in section else section["f1"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[], help="metrics.json path.")
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--pattern", default="seed*/metrics.json")
    parser.add_argument("--group", default="overall")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    paths = list(args.run)
    if args.runs_root is not None:
        paths.extend(sorted(Path(p) for p in glob.glob(str(args.runs_root / args.pattern))))
    if not paths:
        parser.error("provide --run or --runs-root")
    values = [_load(Path(path), args.group) for path in paths]
    summary = {
        "group": args.group,
        "runs": [str(path) for path in paths],
        "values": values,
        "mean_micro_f1": float(np.mean(values)),
        "std_micro_f1": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min_micro_f1": float(np.min(values)),
        "max_micro_f1": float(np.max(values)),
        "seeds": len(values),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
