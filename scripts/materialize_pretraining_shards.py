#!/usr/bin/env python3
"""Materialize a fixed-sequence parquet column as mmap-friendly NPY shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from flowatom.pretraining import materialize_shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Pretraining parquet.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--column", choices=("signed_length", "payload_length", "direction"),
        default="signed_length",
    )
    parser.add_argument("--rows", type=int, default=0, help="Optional uniform subsample size.")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    manifest = materialize_shards(
        args.input,
        args.output_dir,
        column=args.column,
        rows=args.rows,
        seed=args.seed,
    )
    print(json.dumps({key: manifest[key] for key in ("rows", "input_length", "shards")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
