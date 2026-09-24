#!/usr/bin/env python3
"""Generate a small synthetic dataset for smoke tests and examples.

The synthetic traces mimic the FlowAtom data schema: several monitored
websites produce distinguishable packet-length patterns, every visit trace
contains multiple flows, and background traces are unmonitored. It is *not* a
substitute for the real datasets but lets the full pipeline run end to end on
a CPU in seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _clamp(values: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.clip(np.rint(values), low, high).astype(np.int64)


def _make_flow(rng: np.random.Generator, downlink: float, uplink: float, *, phase: int,
               min_packets: int, max_packets: int, min_nonzero: int):
    n_packets = int(rng.integers(min_packets, max_packets + 1))
    directions = np.where(((np.arange(n_packets) + phase) % 3) == 0, -1, 1).astype(np.int64)
    payloads = np.where(
        directions > 0,
        _clamp(rng.normal(downlink, max(10.0, downlink * 0.15), n_packets), 1, 1500),
        _clamp(rng.normal(uplink, max(5.0, uplink * 0.2), n_packets), 1, 1500),
    )
    zero_mask = rng.random(n_packets) < 0.05
    payloads[zero_mask] = 0
    if int((payloads > 0).sum()) < min_nonzero:
        payloads[:min_nonzero] = np.maximum(payloads[:min_nonzero], 1)
    timestamps = np.cumsum(rng.random(n_packets) * 0.02).astype(np.float64)
    return payloads, directions, timestamps


def _site_profile(site: int) -> tuple:
    return 220.0 + 45.0 * site, 45.0 + 6.0 * site, site % 3


def build_traces(
    rng: np.random.Generator,
    *,
    labels,
    traces_per_label: int,
    flows_range,
    min_packets: int,
    max_packets: int,
    min_nonzero: int,
    split: str,
    first_trace_id: int,
):
    rows = []
    trace_id = first_trace_id
    for label in labels:
        downlink, uplink, phase = _site_profile(int(label))
        for order in range(traces_per_label):
            n_flows = int(rng.integers(flows_range[0], flows_range[1] + 1))
            payload_flows, direction_flows, timestamp_flows = [], [], []
            for _ in range(n_flows):
                payload, direction, timestamp = _make_flow(
                    rng,
                    downlink,
                    uplink,
                    phase=phase,
                    min_packets=min_packets,
                    max_packets=max_packets,
                    min_nonzero=min_nonzero,
                )
                payload_flows.append(payload.tolist())
                direction_flows.append(direction.tolist())
                timestamp_flows.append(timestamp.tolist())
            rows.append(
                {
                    "trace_id": trace_id,
                    "label": int(label),
                    "site": f"site-{int(label):03d}",
                    "split": split,
                    "test_order": order,
                    "payload_flows": payload_flows,
                    "direction_flows": direction_flows,
                    "timestamp_flows": timestamp_flows,
                }
            )
            trace_id += 1
    return pd.DataFrame(rows), trace_id


def build_background_traces(
    rng: np.random.Generator,
    *,
    count: int,
    flows_range,
    min_packets: int,
    max_packets: int,
    min_nonzero: int,
    first_trace_id: int,
):
    rows = []
    trace_id = first_trace_id
    for index in range(count):
        downlink = float(rng.uniform(60.0, 1400.0))
        uplink = float(rng.uniform(20.0, 200.0))
        phase = int(rng.integers(0, 3))
        n_flows = int(rng.integers(flows_range[0], flows_range[1] + 1))
        payload_flows, direction_flows, timestamp_flows = [], [], []
        for _ in range(n_flows):
            payload, direction, timestamp = _make_flow(
                rng, downlink, uplink, phase=phase,
                min_packets=min_packets, max_packets=max_packets, min_nonzero=min_nonzero,
            )
            payload_flows.append(payload.tolist())
            direction_flows.append(direction.tolist())
            timestamp_flows.append(timestamp.tolist())
        rows.append(
            {
                "trace_id": trace_id,
                "label": -1,
                "site": f"background-{index:04d}",
                "split": "test",
                "test_order": index,
                "payload_flows": payload_flows,
                "direction_flows": direction_flows,
                "timestamp_flows": timestamp_flows,
            }
        )
        trace_id += 1
    return pd.DataFrame(rows), trace_id


def build_pretraining_flows(
    rng: np.random.Generator,
    *,
    count: int,
    input_length: int,
    min_nonzero: int,
):
    lengths = np.zeros((count, input_length), dtype=np.int32)
    payloads = np.zeros((count, input_length), dtype=np.int32)
    directions = np.zeros((count, input_length), dtype=np.int32)
    for index in range(count):
        downlink = float(rng.uniform(50.0, 1400.0))
        uplink = float(rng.uniform(20.0, 220.0))
        phase = int(rng.integers(0, 3))
        payload, direction, _ = _make_flow(
            rng, downlink, uplink, phase=phase,
            min_packets=min_nonzero + 2, max_packets=input_length, min_nonzero=min_nonzero,
        )
        clipped = min(len(payload), input_length)
        payloads[index, :clipped] = payload[:clipped]
        directions[index, :clipped] = direction[:clipped]
        lengths[index, :clipped] = payload[:clipped] * direction[:clipped]
    return pd.DataFrame(
        {
            "signed_length": [row.tolist() for row in lengths],
            "payload_length": [row.tolist() for row in payloads],
            "direction": [row.tolist() for row in directions],
        }
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sites", type=int, default=20)
    parser.add_argument("--train-traces", type=int, default=8)
    parser.add_argument("--test-traces", type=int, default=4)
    parser.add_argument("--background-traces", type=int, default=60)
    parser.add_argument("--min-flows", type=int, default=1)
    parser.add_argument("--max-flows", type=int, default=4)
    parser.add_argument("--min-packets", type=int, default=6)
    parser.add_argument("--max-packets", type=int, default=40)
    parser.add_argument("--input-length", type=int, default=32)
    parser.add_argument("--min-nonzero-packets", type=int, default=3)
    parser.add_argument("--pretraining-flows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(args.sites))
    rng = np.random.default_rng(args.seed)
    flows_range = (args.min_flows, max(args.min_flows, args.max_flows))

    train_frame, next_id = build_traces(
        rng, labels=labels, traces_per_label=args.train_traces, flows_range=flows_range,
        min_packets=args.min_packets, max_packets=args.max_packets,
        min_nonzero=args.min_nonzero_packets, split="train", first_trace_id=1,
    )
    test_frame, next_id = build_traces(
        rng, labels=labels, traces_per_label=args.test_traces, flows_range=flows_range,
        min_packets=args.min_packets, max_packets=args.max_packets,
        min_nonzero=args.min_nonzero_packets, split="test", first_trace_id=next_id,
    )
    traces = pd.concat([train_frame, test_frame], ignore_index=True)
    traces_path = args.output_dir / "traces.parquet"
    traces.to_parquet(traces_path, index=False)

    background, _ = build_background_traces(
        rng, count=args.background_traces, flows_range=flows_range,
        min_packets=args.min_packets, max_packets=args.max_packets,
        min_nonzero=args.min_nonzero_packets, first_trace_id=next_id,
    )
    background_path = args.output_dir / "background_traces.parquet"
    background.to_parquet(background_path, index=False)

    pretraining = build_pretraining_flows(
        rng, count=args.pretraining_flows, input_length=args.input_length,
        min_nonzero=args.min_nonzero_packets,
    )
    pretraining_path = args.output_dir / "pretraining.parquet"
    pretraining.to_parquet(pretraining_path, index=False)

    summary = {
        "traces": str(traces_path),
        "background_traces": str(background_path),
        "pretraining": str(pretraining_path),
        "sites": args.sites,
        "train_traces": int((traces.split == "train").sum()),
        "test_traces": int((traces.split == "test").sum()),
        "background_trace_count": len(background),
        "pretraining_flows": len(pretraining),
        "input_length": args.input_length,
        "minimum_nonzero_payload_packets": args.min_nonzero_packets,
        "seed": args.seed,
    }
    (args.output_dir / "synthetic_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
