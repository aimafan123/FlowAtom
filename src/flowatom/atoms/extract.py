"""Extract per-trace Atom responses from raw packet sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

from flowatom.atoms.table import TraceAtomError, TraceAtomTable
from flowatom.atoms.vocabulary import AtomVocabulary
from flowatom.encoding import encode_batch
from flowatom.representation import REPRESENTATIONS, flow_representation

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ExtractionConfig:
    """Parameters controlling flow encoding and Atom-response extraction."""

    input_length: int = 300
    batch_size: int = 2048
    confidence_threshold: float = 0.0
    representation: str = "signed_payload_length"
    minimum_nonzero_payload_packets: int = 10
    max_flows_per_trace: int = 0
    flow_sampling_seed: int = 2025
    device: str = "cpu"


def _flush(
    sequences: List[np.ndarray],
    trace_indices: List[int],
    encoder,
    vocabulary: AtomVocabulary,
    peak: np.ndarray,
    device: torch.device,
    confidence_threshold: float,
) -> int:
    if not sequences:
        return 0
    probabilities = vocabulary.predict_proba(encode_batch(sequences, encoder, device))
    retained = probabilities.max(axis=1) >= float(confidence_threshold)
    if np.any(retained):
        np.maximum.at(
            peak,
            np.asarray(trace_indices, dtype=np.int64)[retained],
            probabilities[retained].astype(np.float32),
        )
    sequences.clear()
    trace_indices.clear()
    return int(retained.sum())


def extract_trace_atoms(
    frame,
    encoder,
    vocabulary: AtomVocabulary,
    config: ExtractionConfig = ExtractionConfig(),
) -> TraceAtomTable:
    """Encode every flow and keep the maximum Atom response per trace.

    The returned table stores ``peak[t, a]``, the maximum response of Atom
    ``a`` over the retained flows of trace ``t``. Website labels are carried
    through only for downstream window construction.
    """

    if config.batch_size <= 0 or config.input_length <= 0:
        raise TraceAtomError("batch_size and input_length must be positive")
    if config.representation not in REPRESENTATIONS:
        raise TraceAtomError(f"unsupported representation: {config.representation}")
    if config.max_flows_per_trace < 0:
        raise TraceAtomError("max_flows_per_trace cannot be negative")
    if "direction_flows" not in frame and config.representation in {
        "signed_payload_length",
        "direction_only",
    }:
        raise TraceAtomError(f"{config.representation} requires direction_flows")

    device = torch.device(config.device)
    encoder.to(device).eval()
    trace_count = len(frame)
    trace_ids = frame["trace_id"].to_numpy(dtype=np.int64)
    labels = frame["label"].to_numpy(dtype=np.int64)
    if len(np.unique(trace_ids)) != len(trace_ids):
        raise TraceAtomError("trace_id values must be unique")

    peak = np.zeros((trace_count, vocabulary.atom_count), dtype=np.float32)
    flow_count = np.zeros(trace_count, dtype=np.float32)
    sequences: List[np.ndarray] = []
    trace_indices: List[int] = []
    retained_total = 0
    total_flows = 0
    direction_column = frame.get("direction_flows")

    for trace_index, payloads in enumerate(frame["payload_flows"]):
        directions = None if direction_column is None else direction_column.iloc[trace_index]
        if directions is not None and len(payloads) != len(directions):
            raise TraceAtomError(f"trace {trace_index} has unaligned flow collections")
        flow_indices = np.arange(len(payloads), dtype=np.int64)
        if config.max_flows_per_trace and len(flow_indices) > config.max_flows_per_trace:
            trace_id = int(trace_ids[trace_index])
            rng = np.random.default_rng(config.flow_sampling_seed + trace_id)
            flow_indices = np.sort(
                rng.choice(flow_indices, size=config.max_flows_per_trace, replace=False)
            )
        flow_count[trace_index] = len(flow_indices)
        total_flows += len(flow_indices)
        for flow_index in flow_indices:
            payload = payloads[int(flow_index)]
            direction = (
                [1] * len(payload)
                if directions is None
                else directions[int(flow_index)]
            )
            sequence = flow_representation(
                payload,
                direction,
                representation=config.representation,
                input_length=config.input_length,
                min_payload_packets=config.minimum_nonzero_payload_packets,
            )
            if sequence is None:
                raise TraceAtomError(
                    f"trace {trace_index} flow {int(flow_index)} violates "
                    "minimum_nonzero_payload_packets"
                )
            sequences.append(sequence)
            trace_indices.append(trace_index)
            if len(sequences) >= config.batch_size:
                retained_total += _flush(
                    sequences,
                    trace_indices,
                    encoder,
                    vocabulary,
                    peak,
                    device,
                    config.confidence_threshold,
                )
    retained_total += _flush(
        sequences,
        trace_indices,
        encoder,
        vocabulary,
        peak,
        device,
        config.confidence_threshold,
    )

    table = TraceAtomTable(
        trace_ids=trace_ids,
        labels=labels,
        peak=peak,
        flow_count=flow_count,
        confidence_threshold=float(config.confidence_threshold),
        retained_flow_count=int(retained_total),
        total_flow_count=int(total_flows),
        representation=config.representation,
        input_length=int(config.input_length),
    )
    table.validate()
    return table
