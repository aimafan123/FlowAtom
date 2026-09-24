#!/usr/bin/env python3
"""Build deterministic closed-world window specifications from traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from flowatom.config import load_config, section
from flowatom.artifacts import file_sha256
from flowatom.data import (
    build_closed_world_specs,
    load_trace_frame,
    save_specs,
    validate_specs,
    validate_trace_frame,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--validation-samples", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    closed = section(config, "closed_world")
    test_counts = {int(key): int(value) for key, value in dict(closed["test_samples"]).items()}
    frame = load_trace_frame(args.traces)
    validate_trace_frame(frame)
    specs = build_closed_world_specs(
        frame,
        seed=args.seed,
        train_samples=args.train_samples
        if args.train_samples is not None
        else int(closed["train_samples"]),
        validation_samples=args.validation_samples
        if args.validation_samples is not None
        else int(closed["validation_samples"]),
        test_counts=test_counts,
    )
    specs["trace_dataset_sha256"] = file_sha256(args.traces)
    summary = validate_specs(specs, frame)
    save_specs(specs, args.output)
    print(json.dumps({"output": str(args.output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
