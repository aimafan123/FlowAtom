"""Tests for window aggregation and standardization."""

from pathlib import Path

import numpy as np
import pytest

from flowatom.atoms import TraceAtomTable
from flowatom.window import (
    FeatureStandardizer,
    WindowFeatureError,
    materialize_windows,
    window_representation,
)


def _table():
    peak = np.asarray(
        [[0.1, 0.9, 0.2], [0.7, 0.3, 0.4], [0.2, 0.2, 0.8]], dtype=np.float32
    )
    return TraceAtomTable(
        trace_ids=np.asarray([10, 11, 12]),
        labels=np.asarray([0, 1, 2]),
        peak=peak,
        flow_count=np.asarray([1, 2, 1], dtype=np.float32),
    )


def test_window_representation_is_permutation_invariant_max_pool():
    table = _table()
    left = window_representation([10, 12], table)
    right = window_representation([12, 10], table)
    assert np.array_equal(left, right)
    assert np.allclose(left, [0.2, 0.9, 0.8])


def test_window_representation_rejects_unknown_trace():
    with pytest.raises(WindowFeatureError):
        window_representation([999], _table())


def test_materialize_windows_builds_targets():
    specs = [
        {"websites_num": 2, "labels": [0, 2], "trace_ids": [10, 12]},
        {"websites_num": 1, "labels": [1], "trace_ids": [11]},
    ]
    features, targets = materialize_windows(specs, _table(), labels=[0, 1, 2])
    assert features.shape == (2, 3)
    assert targets.tolist() == [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]


def test_materialize_windows_rejects_unknown_label():
    specs = [{"websites_num": 1, "labels": [7], "trace_ids": [10]}]
    with pytest.raises(WindowFeatureError):
        materialize_windows(specs, _table(), labels=[0, 1, 2])


def test_feature_standardizer_round_trip(tmp_path: Path):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(10, 4)).astype(np.float32)
    standardizer = FeatureStandardizer.fit(features)
    transformed = standardizer.transform(features)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-5)
    path = standardizer.save(tmp_path / "standardizer.npz")
    loaded = FeatureStandardizer.load(path)
    assert np.allclose(loaded.transform(features), transformed)


def test_feature_standardizer_handles_constant_dimension():
    features = np.ones((4, 2), dtype=np.float32)
    standardizer = FeatureStandardizer.fit(features)
    assert standardizer.scale.tolist() == [1.0, 1.0]
