#!/usr/bin/env python3
"""Pretrain the FlowAtom flow encoder with MoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from flowatom.config import load_config, section
from flowatom.pretraining import (
    MoCoConfig,
    ShardedSequenceDataset,
    materialize_shards,
    train_moco,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Shard directory with manifest.json.")
    source.add_argument("--pretraining-parquet", type=Path, help="Raw fixed-sequence parquet.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, help="Where to materialize shards.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--column", choices=("signed_length", "payload_length", "direction"), default="signed_length")
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--augmentation", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    config = section(load_config(args.config), "pretraining")
    if args.manifest is not None:
        manifest = args.manifest
        if manifest.is_dir():
            manifest = manifest / "manifest.json"
    else:
        shards_dir = args.shards_dir or (args.output_dir / "shards")
        materialize_shards(
            args.pretraining_parquet, shards_dir, column=args.column, rows=args.rows,
            seed=args.seed if args.seed is not None else int(config["seed"]),
        )
        manifest = shards_dir / "manifest.json"

    dataset = ShardedSequenceDataset(
        manifest, augment=(args.augmentation or config["augmentation"]) != "identity",
        augmentation=args.augmentation or config["augmentation"],
    )
    moco_config = MoCoConfig(
        epochs=args.epochs if args.epochs is not None else int(config["epochs"]),
        batch_size=args.batch_size if args.batch_size is not None else int(config["batch_size"]),
        learning_rate=args.learning_rate if args.learning_rate is not None else float(config["learning_rate"]),
        momentum=float(config["momentum_sgd"]),
        weight_decay=float(config["weight_decay"]),
        schedule=tuple(int(value) for value in config["schedule"]),
        feature_dim=int(config["feature_dim"]),
        queue_size=int(config["queue_size"]),
        moco_momentum=float(config["momentum"]),
        temperature=float(config["temperature"]),
        workers=args.workers if args.workers is not None else int(config["workers"]),
        seed=args.seed if args.seed is not None else int(config["seed"]),
        device=args.device,
    )
    result = train_moco(dataset, args.output_dir, moco_config, resume=args.resume)
    print(json.dumps({"checkpoint": result["checkpoint"], "epochs": len(result["history"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
