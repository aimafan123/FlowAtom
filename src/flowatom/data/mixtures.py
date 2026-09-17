"""Deterministic construction and validation of website-set windows.

A *window* is an unordered set of complete visit traces. Its label is the set
of monitored websites those traces visit. Windows are built offline from
pre-segmented traces, so FlowAtom never needs visit boundaries or
flow-to-website assignments at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np

from flowatom.data.traces import (
    DatasetValidationError,
    trace_labels,
    trace_splits,
)

PathLike = Union[str, Path]
SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_TEST_COUNTS = {1: 200, 2: 100, 3: 80, 4: 60, 5: 40}


def save_specs(specs: Mapping[str, Any], path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(specs, indent=2))
    return path


def load_specs(path: PathLike) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise DatasetValidationError("window specs must be a JSON object")
    return payload


def _allocate_trace_pools(frame, seed: int):
    """Split training traces into train/validation pools per website."""

    rng = np.random.default_rng(seed)
    train, validation = {}, {}
    for label, group in frame[frame.split == "train"].groupby("label"):
        values = group.trace_id.to_numpy(dtype=np.int64).copy()
        rng.shuffle(values)
        cut = max(1, int(len(values) * 0.9))
        train[int(label)] = values[:cut].tolist()
        validation[int(label)] = values[cut:].tolist() or values[-1:].tolist()
    test = {
        int(label): group.sort_values("test_order").trace_id.astype(int).tolist()
        for label, group in frame[frame.split == "test"].groupby("label")
    }
    return train, validation, test


def _generate_windows(pool: Mapping[int, Sequence[int]], counts: Mapping[int, int], rng):
    labels = np.asarray(
        sorted(label for label, traces in pool.items() if traces), dtype=int
    )
    rows = []
    for websites_num, sample_count in counts.items():
        if len(labels) < websites_num:
            raise DatasetValidationError("not enough labels for requested window size")
        for _ in range(int(sample_count)):
            chosen = rng.choice(labels, size=websites_num, replace=False).tolist()
            trace_ids = [int(rng.choice(pool[int(label)])) for label in chosen]
            rows.append(
                {
                    "websites_num": int(websites_num),
                    "labels": sorted(map(int, chosen)),
                    "trace_ids": trace_ids,
                }
            )
    rng.shuffle(rows)
    return rows


def build_closed_world_specs(
    frame,
    *,
    seed: int = 2025,
    train_samples: int = 15000,
    validation_samples: int = 1500,
    test_counts: Mapping[int, int] = DEFAULT_TEST_COUNTS,
) -> Dict[str, Any]:
    """Build source-only closed-world windows with trace-disjoint splits."""

    train_pool, validation_pool, test_pool = _allocate_trace_pools(frame, seed)
    common = sorted(set(train_pool) & set(validation_pool) & set(test_pool))
    if not common:
        raise DatasetValidationError("no website appears in all three trace pools")
    train_pool = {label: train_pool[label] for label in common}
    validation_pool = {label: validation_pool[label] for label in common}
    test_pool = {label: test_pool[label] for label in common}
    rng = np.random.default_rng(seed)
    payload = {
        "seed": int(seed),
        "labels": common,
        "train": _generate_windows(
            train_pool, {size: train_samples // 5 for size in range(1, 6)}, rng
        ),
        "validation": _generate_windows(
            validation_pool,
            {size: validation_samples // 5 for size in range(1, 6)},
            rng,
        ),
        "test": _generate_windows(test_pool, test_counts, rng),
        "protocol": {
            "source_only_training": True,
            "test_used_for_selection": False,
            "trace_disjoint": True,
        },
    }
    return payload


def build_open_world_specs(
    monitored_frame,
    background_trace_ids: Sequence[int],
    labels: Sequence[int],
    *,
    seed: int = 2025,
    samples_per_cell: int = 500,
    monitored_counts: Sequence[int] = (1, 2, 3, 4, 5),
    background_counts: Sequence[int] = (1, 2, 3, 4, 5),
) -> Dict[str, Any]:
    """Build target-present open-world windows with background traces."""

    monitored = monitored_frame[monitored_frame.split == "test"]
    labels = sorted(int(value) for value in labels)
    pool = {
        int(label): group.trace_id.astype(int).tolist()
        for label, group in monitored[monitored.label.isin(labels)].groupby("label")
    }
    missing = [label for label in labels if not pool.get(label)]
    if missing:
        raise DatasetValidationError(f"no test traces for monitored labels {missing}")
    background_ids = [int(value) for value in background_trace_ids]
    overlap = set(background_ids) & set(int(value) for value in monitored["trace_id"])
    if overlap:
        raise DatasetValidationError(
            f"background trace ids overlap monitored trace ids: {len(overlap)}"
        )
    monitored_counts = sorted({int(value) for value in monitored_counts})
    background_counts = sorted({int(value) for value in background_counts})
    if min(monitored_counts) < 1 or max(monitored_counts) > len(labels):
        raise DatasetValidationError("monitored counts must lie in [1, len(labels)]")
    if min(background_counts) < 0:
        raise DatasetValidationError("background counts must be non-negative")
    if max(background_counts) > len(background_ids):
        raise DatasetValidationError("background count exceeds the trace pool")

    rng = np.random.default_rng(seed)
    rows = []
    cells = []
    for monitored_count in monitored_counts:
        for background_count in background_counts:
            cells.append((monitored_count, background_count))
            for _ in range(int(samples_per_cell)):
                chosen = rng.choice(labels, size=monitored_count, replace=False).tolist()
                monitored_ids = [int(rng.choice(pool[int(label)])) for label in chosen]
                background_chosen = (
                    rng.choice(background_ids, size=background_count, replace=False)
                    .astype(int)
                    .tolist()
                    if background_count
                    else []
                )
                rows.append(
                    {
                        "websites_num": int(monitored_count),
                        "monitored_websites_num": int(monitored_count),
                        "background_traces_num": int(background_count),
                        "labels": sorted(map(int, chosen)),
                        "trace_ids": monitored_ids + background_chosen,
                    }
                )
    rng.shuffle(rows)
    return {
        "seed": int(seed),
        "labels": labels,
        "train": [],
        "validation": [],
        "test": rows,
        "protocol": {
            "samples_per_cell": int(samples_per_cell),
            "monitored_counts": monitored_counts,
            "background_counts": background_counts,
            "cells": [list(cell) for cell in cells],
            "background_sampling_unit": "trace",
            "background_used_for_training": False,
            "background_used_for_selection": False,
            "decoder_frozen_from_source_validation": True,
        },
    }


def _validate_window(
    item: Mapping[str, Any],
    labels: set,
    labels_by_trace: Mapping[int, int],
    background_ids: set,
) -> Sequence[int]:
    websites_num = int(item["websites_num"])
    item_labels = [int(value) for value in item["labels"]]
    trace_ids = [int(value) for value in item["trace_ids"]]
    background_count = int(
        item.get("background_traces_num", item.get("background_websites_num", 0))
    )
    if websites_num != len(item_labels):
        raise DatasetValidationError("websites_num must equal the label count")
    if len(trace_ids) != len(item_labels) + background_count:
        raise DatasetValidationError(
            "trace count must equal monitored labels plus background traces"
        )
    if len(set(item_labels)) != len(item_labels):
        raise DatasetValidationError("a window cannot repeat a website label")
    if len(set(trace_ids)) != len(trace_ids):
        raise DatasetValidationError("a window cannot repeat a trace")
    if not set(item_labels).issubset(labels):
        raise DatasetValidationError("window contains an unknown label")
    observed_monitored = set()
    for trace_id in trace_ids:
        if trace_id in labels_by_trace:
            observed_monitored.add(labels_by_trace[trace_id])
        elif trace_id not in background_ids:
            raise DatasetValidationError(
                f"window references unknown trace_id {trace_id}"
            )
    # Background traces are unmonitored and never appear in the window label.
    observed_monitored &= labels
    if observed_monitored != set(item_labels):
        raise DatasetValidationError("window labels do not match the trace labels")
    return trace_ids


def validate_specs(
    specs: Mapping[str, Any], frame, background_trace_ids: Sequence[int] = ()
) -> Dict[str, Any]:
    """Validate window specs, split roles and cross-split trace leakage."""

    required = {"seed", "labels", "train", "validation", "test"}
    missing = sorted(required - set(specs))
    if missing:
        raise DatasetValidationError(f"missing window-spec fields: {missing}")
    labels = [int(value) for value in specs["labels"]]
    if len(labels) != len(set(labels)):
        raise DatasetValidationError("window labels must be unique")
    labels_by_trace = trace_labels(frame)
    splits_by_trace = trace_splits(frame)
    background_ids = {int(value) for value in background_trace_ids}

    used_traces: Dict[str, set] = {}
    counts: Dict[str, int] = {}
    distribution: Dict[str, Dict[str, int]] = {}
    for split_name in SPLIT_NAMES:
        items = specs[split_name]
        if not isinstance(items, list):
            raise DatasetValidationError(f"{split_name} specs must be a list")
        if split_name != "test" and not items:
            used_traces[split_name] = set()
            counts[split_name] = 0
            distribution[split_name] = {}
            continue
        if not items:
            raise DatasetValidationError("test specs must be non-empty")
        used = set()
        counts_by_size = {str(size): 0 for size in range(1, 6)}
        expected_trace_split = "test" if split_name == "test" else "train"
        for item in items:
            trace_ids = _validate_window(item, set(labels), labels_by_trace, background_ids)
            for trace_id in trace_ids:
                if trace_id in labels_by_trace and labels_by_trace[trace_id] in set(labels):
                    if splits_by_trace[trace_id] != expected_trace_split:
                        raise DatasetValidationError(
                            f"{split_name} uses trace {trace_id} from "
                            f"{splits_by_trace[trace_id]}"
                        )
            used.update(trace_ids)
            size_key = str(int(item["websites_num"]))
            if size_key not in counts_by_size:
                raise DatasetValidationError("websites_num must be between 1 and 5")
            counts_by_size[size_key] += 1
        used_traces[split_name] = used
        counts[split_name] = len(items)
        distribution[split_name] = counts_by_size

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = used_traces.get(left, set()) & used_traces.get(right, set())
        if overlap:
            raise DatasetValidationError(
                f"trace leakage between {left} and {right}: {len(overlap)}"
            )
    return {
        "seed": int(specs["seed"]),
        "labels": len(labels),
        "windows": counts,
        "unique_traces": {key: len(value) for key, value in used_traces.items()},
        "count_distribution": distribution,
    }
