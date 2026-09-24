"""Per-trace Atom responses with a verifiable feature contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, Union

import numpy as np

from flowatom.artifacts import require_equal

PathLike = Union[str, Path]
CONTRACT_FIELDS = {
    "encoder_sha256", "vocabulary_sha256", "representation", "input_length",
    "minimum_nonzero_payload_packets", "confidence_threshold",
    "max_flows_per_trace", "flow_sampling_seed",
}


class TraceAtomError(ValueError):
    """Raised when a trace-atom table violates its contract."""


@dataclass(frozen=True)
class TraceAtomTable:
    """Per-trace maxima; flow_count counts flows that actually contribute.

    Traces with no eligible or confident flows are retained with zero peaks.
    The contract binds the columns to a particular encoder, vocabulary and
    extraction policy. atom_fit records the vocabulary's training trace pool.
    """

    trace_ids: np.ndarray
    labels: np.ndarray
    peak: np.ndarray
    flow_count: np.ndarray
    confidence_threshold: float = 0.0
    retained_flow_count: int = 0
    total_flow_count: int = 0
    representation: str = "signed_payload_length"
    input_length: int = 300
    short_flow_count: int = 0
    source_sha256: str = ""
    contract: dict = field(default_factory=dict)
    atom_fit: dict = field(default_factory=dict)

    @property
    def trace_count(self) -> int:
        return int(len(self.trace_ids))

    @property
    def atom_count(self) -> int:
        return int(self.peak.shape[1])

    @cached_property
    def index_by_trace(self) -> Dict[int, int]:
        return {int(value): index for index, value in enumerate(self.trace_ids)}

    def validate(self) -> None:
        if self.peak.ndim != 2 or self.peak.shape[1] == 0:
            raise TraceAtomError("peak responses must have shape [traces, atoms>0]")
        rows = len(self.trace_ids)
        if self.trace_ids.shape != (rows,) or len(np.unique(self.trace_ids)) != rows:
            raise TraceAtomError("trace_ids must be a unique vector")
        if self.labels.shape != (rows,) or self.flow_count.shape != (rows,):
            raise TraceAtomError("trace metadata shapes are inconsistent")
        if self.peak.shape[0] != rows:
            raise TraceAtomError("peak responses must align with trace_ids")
        if not np.all(np.isfinite(self.peak)) or np.any((self.peak < 0) | (self.peak > 1)):
            raise TraceAtomError("peak responses must be finite probabilities")
        if not np.all(np.isfinite(self.flow_count)) or np.any(self.flow_count < 0):
            raise TraceAtomError("flow counts must be finite and non-negative")
        if np.any(self.flow_count != np.floor(self.flow_count)):
            raise TraceAtomError("flow counts must be integers")
        if int(self.flow_count.sum()) != self.retained_flow_count:
            raise TraceAtomError("retained_flow_count must equal the per-trace count sum")
        if self.short_flow_count < 0 or self.total_flow_count < self.retained_flow_count + self.short_flow_count:
            raise TraceAtomError("inconsistent raw/short/retained flow counts")
        if np.any(self.peak[self.flow_count == 0] != 0):
            raise TraceAtomError("zero-flow traces must have zero Atom responses")
        if not CONTRACT_FIELDS.issubset(self.contract) or not self.atom_fit:
            raise TraceAtomError("cache has no complete provenance; rebuild trace Atom responses")
        if not self.contract["encoder_sha256"] or not self.contract["vocabulary_sha256"]:
            raise TraceAtomError("cache must identify its encoder and vocabulary")
        require_equal(self.contract["input_length"], self.input_length, "cache input length")
        require_equal(self.contract["representation"], self.representation, "cache representation")
        require_equal(self.contract["confidence_threshold"], self.confidence_threshold, "cache confidence threshold")

    def save(self, path: PathLike) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 2, "source_sha256": self.source_sha256, "contract": self.contract, "atom_fit": self.atom_fit,
            "retained_flow_count": self.retained_flow_count,
            "total_flow_count": self.total_flow_count, "short_flow_count": self.short_flow_count,
        }
        np.savez_compressed(
            path, trace_ids=self.trace_ids.astype(np.int64), labels=self.labels.astype(np.int64),
            peak=self.peak.astype(np.float32), flow_count=self.flow_count.astype(np.int64),
            metadata=np.asarray([json.dumps(metadata)]),
        )
        return path

    @classmethod
    def load(cls, path: PathLike) -> "TraceAtomTable":
        with np.load(Path(path), allow_pickle=False) as saved:
            required = {"trace_ids", "labels", "peak", "flow_count", "metadata"}
            if not required.issubset(saved.files):
                raise TraceAtomError("legacy or incomplete cache; rebuild trace Atom responses")
            metadata = json.loads(str(saved["metadata"][0]))
            if metadata.get("schema_version") != 2:
                raise TraceAtomError("legacy cache has no provenance; rebuild trace Atom responses")
            contract = metadata["contract"]
            table = cls(
                trace_ids=saved["trace_ids"].copy(), labels=saved["labels"].copy(),
                peak=saved["peak"].copy(), flow_count=saved["flow_count"].copy(),
                confidence_threshold=contract["confidence_threshold"],
                representation=contract["representation"], input_length=contract["input_length"],
                retained_flow_count=metadata["retained_flow_count"],
                total_flow_count=metadata["total_flow_count"], short_flow_count=metadata["short_flow_count"],
                contract=contract, atom_fit=metadata["atom_fit"],
                source_sha256=metadata.get("source_sha256", ""),
            )
        table.validate()
        return table


def concatenate_trace_atom_tables(tables: list) -> TraceAtomTable:
    if not tables:
        raise TraceAtomError("at least one trace-atom table is required")
    first = tables[0]
    for table in tables:
        table.validate()
        require_equal(table.atom_count, first.atom_count, "cache Atom count")
        require_equal(table.contract, first.contract, "cache feature contract")
        require_equal(table.atom_fit, first.atom_fit, "cache Atom training provenance")
    combined = TraceAtomTable(
        trace_ids=np.concatenate([table.trace_ids for table in tables]),
        labels=np.concatenate([table.labels for table in tables]),
        peak=np.concatenate([table.peak for table in tables]),
        flow_count=np.concatenate([table.flow_count for table in tables]),
        confidence_threshold=first.confidence_threshold,
        retained_flow_count=sum(table.retained_flow_count for table in tables),
        total_flow_count=sum(table.total_flow_count for table in tables),
        short_flow_count=sum(table.short_flow_count for table in tables),
        representation=first.representation, input_length=first.input_length,
        contract=first.contract, atom_fit=first.atom_fit,
    )
    combined.validate()
    return combined
