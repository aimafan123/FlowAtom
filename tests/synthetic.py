"""Synthetic in-memory trace frames for tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_flow(rng, mean, packets, min_nonzero):
    payload = np.clip(np.rint(rng.normal(mean, max(5.0, mean * 0.2), packets)), 1, 1500).astype(int)
    payload[rng.random(packets) < 0.1] = 0
    if int((payload > 0).sum()) < min_nonzero:
        payload[:min_nonzero] = np.maximum(payload[:min_nonzero], 1)
    direction = rng.choice([-1, 1], size=packets)
    timestamp = np.cumsum(rng.random(packets) * 0.01)
    return payload.tolist(), direction.tolist(), timestamp.tolist()


def offset_trace_ids(frame, offset: int = 100000):
    """Return a copy whose trace ids are shifted to stay globally unique."""

    frame = frame.copy()
    frame["trace_id"] = frame["trace_id"].astype(int) + int(offset)
    return frame


def make_traces(
    *,
    num_sites: int = 4,
    train_per_site: int = 3,
    test_per_site: int = 2,
    background: int = 0,
    flows_per_trace: int = 2,
    packets: int = 8,
    min_nonzero: int = 2,
    seed: int = 0,
):
    """Create a small in-memory trace frame following the FlowAtom schema."""

    rng = np.random.default_rng(seed)
    rows = []
    trace_id = 1
    for split, count in (("train", train_per_site), ("test", test_per_site)):
        for site in range(num_sites):
            mean = 200.0 + 90.0 * site
            for order in range(count):
                payload_flows, direction_flows, timestamp_flows = [], [], []
                for _ in range(flows_per_trace):
                    payload, direction, timestamp = _make_flow(rng, mean, packets, min_nonzero)
                    payload_flows.append(payload)
                    direction_flows.append(direction)
                    timestamp_flows.append(timestamp)
                rows.append(
                    {
                        "trace_id": trace_id,
                        "label": site,
                        "site": f"site-{site}",
                        "split": split,
                        "test_order": order,
                        "payload_flows": payload_flows,
                        "direction_flows": direction_flows,
                        "timestamp_flows": timestamp_flows,
                    }
                )
                trace_id += 1
    for index in range(background):
        payload_flows, direction_flows, timestamp_flows = [], [], []
        for _ in range(flows_per_trace):
            payload, direction, timestamp = _make_flow(rng, 300.0, packets, min_nonzero)
            payload_flows.append(payload)
            direction_flows.append(direction)
            timestamp_flows.append(timestamp)
        rows.append(
            {
                "trace_id": trace_id,
                "label": -1,
                "site": f"background-{index}",
                "split": "test",
                "test_order": index,
                "payload_flows": payload_flows,
                "direction_flows": direction_flows,
                "timestamp_flows": timestamp_flows,
            }
        )
        trace_id += 1
    return pd.DataFrame(rows)
