"""Trace dataset schema, loading and validation.

A FlowAtom trace dataset is a parquet file with one row per visit trace. Each
row stores a nested list of flows; every flow is a packet sequence. The schema
is intentionally minimal: identity information such as IP addresses, DNS names
and SNI is never part of the model input.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Union

import numpy as np

from flowatom.representation import REPRESENTATIONS, flow_representation

PathLike = Union[str, Path]
TRACE_COLUMNS = (
    "trace_id",
    "label",
    "site",
    "split",
    "test_order",
    "payload_flows",
    "direction_flows",
)
OPTIONAL_TRACE_COLUMNS = ("timestamp_flows",)
ALLOWED_SPLITS = {"train", "test"}


class DatasetValidationError(ValueError):
    """Raised when data violates the FlowAtom trace contract."""


@dataclass(frozen=True)
class TraceDatasetSummary:
    rows: int
    traces: int
    flows: int
    labels: int
    split_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_trace_frame(path: PathLike, columns: Optional[Sequence[str]] = None):
    """Load a trace parquet file."""

    import pandas as pd

    selected = list(columns) if columns is not None else list(TRACE_COLUMNS)
    return pd.read_parquet(Path(path), columns=selected)


def save_trace_frame(frame, path: PathLike) -> Path:
    """Write a trace frame to parquet, creating parent directories."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def validate_trace_frame(frame) -> TraceDatasetSummary:
    """Validate one-row-per-trace data and nested flow alignment."""

    missing = sorted(set(TRACE_COLUMNS) - set(frame.columns))
    if missing:
        raise DatasetValidationError(f"missing trace columns: {missing}")
    if frame.empty:
        raise DatasetValidationError("trace dataset is empty")
    if frame["trace_id"].isna().any() or frame["trace_id"].duplicated().any():
        raise DatasetValidationError("trace_id must be non-null and unique")
    actual_splits = {str(value) for value in frame["split"].unique()}
    if not actual_splits.issubset(ALLOWED_SPLITS):
        raise DatasetValidationError(
            f"unexpected trace splits: {sorted(actual_splits - ALLOWED_SPLITS)}"
        )
    if int(frame.groupby("label")["site"].nunique().max()) != 1:
        raise DatasetValidationError("each label must map to exactly one site")

    flow_count = 0
    for row in frame.itertuples(index=False):
        payloads = row.payload_flows
        directions = row.direction_flows
        if len(payloads) != len(directions):
            raise DatasetValidationError(
                f"trace {row.trace_id} has unaligned flow collections"
            )
        if len(payloads) == 0:
            raise DatasetValidationError(f"trace {row.trace_id} has no flows")
        flow_count += len(payloads)
        for flow_index, (payload, direction) in enumerate(zip(payloads, directions)):
            if len(payload) != len(direction):
                raise DatasetValidationError(
                    f"trace {row.trace_id} flow {flow_index} has unaligned packets"
                )
            if len(payload) == 0:
                raise DatasetValidationError(
                    f"trace {row.trace_id} flow {flow_index} is empty"
                )
    split_counts = {
        str(key): int(value) for key, value in frame["split"].value_counts().items()
    }
    return TraceDatasetSummary(
        rows=int(len(frame)),
        traces=int(frame["trace_id"].nunique()),
        flows=int(flow_count),
        labels=int(frame["label"].nunique()),
        split_counts=split_counts,
    )


def trace_labels(frame) -> Dict[int, int]:
    return {
        int(trace_id): int(label)
        for trace_id, label in zip(frame["trace_id"], frame["label"])
    }


def trace_splits(frame) -> Dict[int, str]:
    return {
        int(trace_id): str(split)
        for trace_id, split in zip(frame["trace_id"], frame["split"])
    }


def iter_flow_sequences(
    frame,
    *,
    split: Optional[str] = None,
    representation: str = "signed_payload_length",
    input_length: int = 300,
    min_payload_packets: int = 10,
    skip_filtered: bool = True,
) -> Iterator[np.ndarray]:
    """Yield one fixed-length sequence per flow of the selected traces."""

    if representation not in REPRESENTATIONS:
        raise DatasetValidationError(f"unsupported representation: {representation}")
    selected = frame if split is None else frame[frame["split"] == split]
    has_direction = "direction_flows" in selected.columns
    for trace_id, payloads, directions in zip(
        selected["trace_id"],
        selected["payload_flows"],
        selected["direction_flows"] if has_direction else [None] * len(selected),
    ):
        if directions is None:
            directions = [[1] * len(payload) for payload in payloads]
        if len(payloads) != len(directions):
            raise DatasetValidationError(f"trace {trace_id} has unaligned flow collections")
        for flow_index, (payload, direction) in enumerate(zip(payloads, directions)):
            sequence = flow_representation(
                payload,
                direction,
                representation=representation,
                input_length=input_length,
                min_payload_packets=min_payload_packets,
            )
            if sequence is None:
                if skip_filtered:
                    continue
                raise DatasetValidationError(
                    f"trace {trace_id} flow {flow_index} violates "
                    "minimum_nonzero_payload_packets"
                )
            yield sequence


def load_json(path: PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
