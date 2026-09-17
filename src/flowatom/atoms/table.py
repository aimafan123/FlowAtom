"""Per-trace Atom response tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Union

import numpy as np

PathLike = Union[str, Path]


class TraceAtomError(ValueError):
    """Raised when a trace-atom table violates its contract."""


@dataclass(frozen=True)
class TraceAtomTable:
    """Per-trace maximum Atom responses.

    ``peak[t, a]`` is the maximum Atom-``a`` response over all retained flows of
    trace ``t``. Max pooling this matrix over the traces of a window yields the
    FlowAtom window representation.
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
        if self.peak.ndim != 2:
            raise TraceAtomError("peak responses must have shape [traces, atoms]")
        rows = len(self.trace_ids)
        if len(np.unique(self.trace_ids)) != rows:
            raise TraceAtomError("trace_ids must be unique")
        if self.labels.shape != (rows,) or self.flow_count.shape != (rows,):
            raise TraceAtomError("trace metadata shapes are inconsistent")
        if self.peak.shape[0] != rows:
            raise TraceAtomError("peak responses must align with trace_ids")
        if not np.all(np.isfinite(self.peak)):
            raise TraceAtomError("peak responses must be finite")

    def save(self, path: PathLike) -> Path:
        """Persist the table as a compressed ``.npz`` archive."""

        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            trace_ids=self.trace_ids.astype(np.int64),
            labels=self.labels.astype(np.int64),
            peak=self.peak.astype(np.float32),
            flow_count=self.flow_count.astype(np.float32),
            confidence_threshold=np.asarray([self.confidence_threshold], dtype=np.float32),
            retained_flow_count=np.asarray([self.retained_flow_count], dtype=np.int64),
            total_flow_count=np.asarray([self.total_flow_count], dtype=np.int64),
            metadata=np.asarray(
                [
                    json.dumps(
                        {
                            "representation": self.representation,
                            "input_length": int(self.input_length),
                        }
                    )
                ]
            ),
        )
        return path

    @classmethod
    def load(cls, path: PathLike) -> "TraceAtomTable":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as saved:
            required = {"trace_ids", "labels", "peak", "flow_count"}
            missing = sorted(required - set(saved.files))
            if missing:
                raise TraceAtomError(f"trace-atom cache misses {missing}")
            metadata = {}
            if "metadata" in saved.files:
                metadata = json.loads(str(saved["metadata"][0]))
            table = cls(
                trace_ids=saved["trace_ids"].copy(),
                labels=saved["labels"].copy(),
                peak=saved["peak"].copy(),
                flow_count=saved["flow_count"].copy(),
                confidence_threshold=float(saved["confidence_threshold"][0])
                if "confidence_threshold" in saved.files
                else 0.0,
                retained_flow_count=int(saved["retained_flow_count"][0])
                if "retained_flow_count" in saved.files
                else int(saved["flow_count"].sum()),
                total_flow_count=int(saved["total_flow_count"][0])
                if "total_flow_count" in saved.files
                else int(saved["flow_count"].sum()),
                representation=str(metadata.get("representation", "signed_payload_length")),
                input_length=int(metadata.get("input_length", 300)),
            )
        table.validate()
        return table


def concatenate_trace_atom_tables(tables: list) -> TraceAtomTable:
    """Combine compatible tables, e.g. monitored and background traces."""

    if not tables:
        raise TraceAtomError("at least one trace-atom table is required")
    for table in tables:
        table.validate()
    atom_count = tables[0].atom_count
    if any(table.atom_count != atom_count for table in tables):
        raise TraceAtomError("trace-atom tables use different vocabularies")
    combined = TraceAtomTable(
        trace_ids=np.concatenate([table.trace_ids for table in tables]),
        labels=np.concatenate([table.labels for table in tables]),
        peak=np.concatenate([table.peak for table in tables]),
        flow_count=np.concatenate([table.flow_count for table in tables]),
        confidence_threshold=tables[0].confidence_threshold,
        retained_flow_count=sum(table.retained_flow_count for table in tables),
        total_flow_count=sum(table.total_flow_count for table in tables),
        representation=tables[0].representation,
        input_length=tables[0].input_length,
    )
    combined.validate()
    return combined
