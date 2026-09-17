"""Tests for window-predictor training and frozen evaluation."""

import json
from pathlib import Path

import numpy as np

from flowatom.atoms import TraceAtomTable, concatenate_trace_atom_tables
from flowatom.data import build_closed_world_specs, build_open_world_specs
from flowatom.training import (
    WindowPredictorConfig,
    evaluate_frozen,
    load_frozen_predictor,
    train_window_predictor,
)

from synthetic import make_traces, offset_trace_ids


def _random_table(frame, atom_count: int, seed: int = 0) -> TraceAtomTable:
    rng = np.random.default_rng(seed)
    return TraceAtomTable(
        trace_ids=frame["trace_id"].to_numpy(dtype=np.int64),
        labels=frame["label"].to_numpy(dtype=np.int64),
        peak=rng.random((len(frame), atom_count)).astype(np.float32),
        flow_count=np.asarray([len(flows) for flows in frame["payload_flows"]], dtype=np.float32),
    )


def test_train_and_frozen_evaluate(tmp_path: Path):
    frame = make_traces(num_sites=5, train_per_site=3, test_per_site=2)
    background = offset_trace_ids(make_traces(num_sites=0, train_per_site=0, test_per_site=0, background=4))
    specs = build_closed_world_specs(frame, seed=0, train_samples=40, validation_samples=20)
    table = _random_table(frame, atom_count=6)

    result = train_window_predictor(
        specs,
        table,
        tmp_path / "run",
        WindowPredictorConfig(
            seed=0, epochs=3, minimum_epochs=1, patience=2, hidden_dim=8, batch_size=8
        ),
    )
    assert 0.0 <= result["test"]["overall"]["micro_f1"] <= 1.0
    assert result["protocol"]["test_used_for_selection"] is False
    assert set(result["test"]) == {"1", "2", "3", "4", "5", "overall"}
    assert (tmp_path / "run" / "metrics.json").is_file()
    assert (tmp_path / "run" / "model.pt").is_file()
    assert (tmp_path / "run" / "standardizer.npz").is_file()

    predictor = load_frozen_predictor(tmp_path / "run")
    closed = evaluate_frozen(predictor, specs, table, mode="closed_world")
    assert "overall" in closed and "groups" in closed

    open_specs = build_open_world_specs(
        frame, background["trace_id"].astype(int).tolist(), specs["labels"],
        seed=0, samples_per_cell=3, monitored_counts=(1, 2), background_counts=(0, 2),
    )
    combined = concatenate_trace_atom_tables([table, _random_table(background, atom_count=6, seed=1)])
    opened = evaluate_frozen(predictor, open_specs, combined, mode="open_world")
    assert "overall" in opened
    assert any("/" in key for key in opened["groups"])


def test_training_saves_json_metrics(tmp_path: Path):
    frame = make_traces(num_sites=5, train_per_site=2, test_per_site=2)
    specs = build_closed_world_specs(frame, seed=1, train_samples=30, validation_samples=15)
    table = _random_table(frame, atom_count=5)
    train_window_predictor(
        specs, table, tmp_path / "run",
        WindowPredictorConfig(seed=1, epochs=2, minimum_epochs=1, patience=2, hidden_dim=8, batch_size=8),
    )
    payload = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert payload["labels"] == [0, 1, 2, 3, 4]
    assert payload["atom_count"] == 5
