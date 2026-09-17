#!/usr/bin/env python3
"""Train the window-level website-set predictor on closed-world windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from flowatom.atoms import TraceAtomTable
from flowatom.config import load_config, section
from flowatom.data import load_specs
from flowatom.training import WindowPredictorConfig, train_window_predictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--atoms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-websites", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    predictor = section(config, "predictor")
    window = section(config, "window")
    if args.seed is None:
        seeds = [int(value) for value in predictor["seeds"]]
        if len(seeds) != 1:
            parser.error(
                "config lists multiple seeds; pass --seed explicitly and run one seed per invocation"
            )
        seed = seeds[0]
    else:
        seed = args.seed
    output_dir = args.output_dir
    result = train_window_predictor(
        load_specs(args.specs),
        TraceAtomTable.load(args.atoms),
        output_dir,
        WindowPredictorConfig(
            seed=seed,
            batch_size=args.batch_size if args.batch_size is not None else int(predictor["batch_size"]),
            epochs=args.epochs if args.epochs is not None else int(predictor["epochs"]),
            patience=args.patience if args.patience is not None else int(predictor["patience"]),
            hidden_dim=args.hidden_dim if args.hidden_dim is not None else int(predictor["hidden_dim"]),
            dropout=args.dropout if args.dropout is not None else float(predictor["dropout"]),
            learning_rate=args.learning_rate
            if args.learning_rate is not None
            else float(predictor["learning_rate"]),
            weight_decay=float(predictor["weight_decay"]),
            max_websites=args.max_websites
            if args.max_websites is not None
            else int(window["max_websites_per_window"]),
            device=args.device,
        ),
    )
    print(json.dumps(result["test"]["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
