"""Shared fixtures for the FlowAtom test suite."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic import make_traces


@pytest.fixture
def tiny_traces():
    return make_traces(num_sites=5)


@pytest.fixture
def tiny_sequences():
    rng = np.random.default_rng(7)
    return rng.integers(-1500, 1500, size=(64, 8)).astype(np.float32)


@pytest.fixture(scope="session")
def xgboost_available():
    pytest.importorskip("xgboost")
    return True
