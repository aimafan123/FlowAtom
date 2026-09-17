#!/usr/bin/env python3
"""Evaluate a frozen FlowAtom decoder on drift or open-world windows.

The decoder and its threshold come from the source-domain validation split.
Target or background traffic is never used for training or for selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from flowatom.atoms import TraceAtomTable, concatenate_trace_atom_tables
from flowatom.data import load_specs
from flowatom.training import evaluate_frozen, load_frozen_predictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--evaluation-specs", type=Path, required=True)
    parser.add_argument(
        "--atoms", type=Path, action="append", required=True,
        help="Trace-atom table; repeat to combine monitored and background caches.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("closed_world", "open_world"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-websites", type=int, default=None)
    args = parser.parse_args()

    predictor = load_frozen_predictor(args.source_run, device="cpu")
    atoms = concatenate_trace_atom_tables([TraceAtomTable.load(path) for path in args.atoms])
    specs = load_specs(args.evaluation_specs)
    result = evaluate_frozen(
        predictor,
        specs,
        atoms,
        mode=args.mode,
        device=args.device,
        max_websites=args.max_websites,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
