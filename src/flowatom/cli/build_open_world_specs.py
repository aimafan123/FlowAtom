#!/usr/bin/env python3
"""Build target-present open-world window specifications.

Windows combine monitored test traces with unmonitored background traces.
Background traffic is used for evaluation only; it is never used for training
or for threshold selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from flowatom.config import load_config, section
from flowatom.data import (
    build_open_world_specs,
    load_specs,
    load_trace_frame,
    save_specs,
    validate_specs,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitored-traces", type=Path, required=True)
    parser.add_argument("--background-traces", type=Path, required=True)
    parser.add_argument("--source-specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--samples-per-cell", type=int, default=None)
    parser.add_argument("--monitored-counts", default=None, help="Comma-separated counts.")
    parser.add_argument("--background-counts", default=None, help="Comma-separated counts.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    open_world = section(config, "open_world")
    monitored_counts = (
        [int(value) for value in args.monitored_counts.split(",")]
        if args.monitored_counts
        else [int(value) for value in open_world["monitored_counts"]]
    )
    background_counts = (
        [int(value) for value in args.background_counts.split(",")]
        if args.background_counts
        else [int(value) for value in open_world["background_counts"]]
    )
    monitored = load_trace_frame(args.monitored_traces)
    background = load_trace_frame(args.background_traces, columns=["trace_id", "label", "site", "split"])
    source = load_specs(args.source_specs)
    specs = build_open_world_specs(
        monitored,
        background["trace_id"].astype(int).tolist(),
        source["labels"],
        seed=args.seed,
        samples_per_cell=args.samples_per_cell
        if args.samples_per_cell is not None
        else int(open_world["samples_per_cell"]),
        monitored_counts=monitored_counts,
        background_counts=background_counts,
    )
    summary = validate_specs(specs, monitored, background_trace_ids=background["trace_id"].astype(int).tolist())
    save_specs(specs, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                **summary,
                "cells": len(specs["protocol"]["cells"]),
                "samples_per_cell": specs["protocol"]["samples_per_cell"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
