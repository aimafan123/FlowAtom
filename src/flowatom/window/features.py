"""Permutation-invariant window representations and multi-label targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple, Union

import numpy as np

from flowatom.atoms.table import TraceAtomTable

PathLike = Union[str, Path]


class WindowFeatureError(ValueError):
    """Raised when window specs or Atom tables violate the feature contract."""


def window_representation(trace_ids: Sequence[int], atoms: TraceAtomTable) -> np.ndarray:
    """Max-pool Atom responses across the traces of one observation window.

    Because ``atoms.peak`` already holds the per-trace maximum over flows, max
    pooling over traces is equivalent to max pooling over every flow in the
    window and is invariant to flow order.
    """

    if not trace_ids:
        raise WindowFeatureError("a window must contain at least one trace")
    index = atoms.index_by_trace
    try:
        indices = [index[int(trace_id)] for trace_id in trace_ids]
    except KeyError as error:
        raise WindowFeatureError(f"unknown trace_id {error.args[0]}") from error
    return atoms.peak[indices].max(axis=0).astype(np.float32)


def materialize_windows(
    specs: Sequence[Mapping],
    atoms: TraceAtomTable,
    labels: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the window feature matrix and its multi-label target matrix."""

    label_index = {int(label): position for position, label in enumerate(labels)}
    features = np.empty((len(specs), atoms.atom_count), dtype=np.float32)
    targets = np.zeros((len(specs), len(labels)), dtype=np.float32)
    for row, spec in enumerate(specs):
        features[row] = window_representation(spec["trace_ids"], atoms)
        for label in spec["labels"]:
            try:
                targets[row, label_index[int(label)]] = 1.0
            except KeyError as error:
                raise WindowFeatureError(f"unknown label {error.args[0]}") from error
    return features, targets


@dataclass(frozen=True)
class FeatureStandardizer:
    """Per-dimension standardization fitted on downstream training windows."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training_features: np.ndarray) -> "FeatureStandardizer":
        values = np.asarray(training_features, dtype=np.float32)
        if values.ndim != 2 or len(values) == 0:
            raise WindowFeatureError("training features must be a non-empty matrix")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise WindowFeatureError("feature dimension does not match standardizer")
        return (values - self.mean) / self.scale

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "FeatureStandardizer":
        with np.load(Path(path)) as saved:
            return cls(mean=saved["mean"].astype(np.float32), scale=saved["scale"].astype(np.float32))
