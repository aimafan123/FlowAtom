"""End-to-end integration test of the FlowAtom pipeline."""

import json
from pathlib import Path

from flowatom.atoms import (
    ExtractionConfig,
    build_atom_vocabulary,
    concatenate_trace_atom_tables,
    extract_trace_atoms,
)
from flowatom.data import (
    build_closed_world_specs,
    build_open_world_specs,
    iter_flow_sequences,
    validate_specs,
)
from flowatom.encoding import embed_sequences
from flowatom.models import DFMiniEncoder
from flowatom.training import (
    WindowPredictorConfig,
    evaluate_frozen,
    load_frozen_predictor,
    train_window_predictor,
)

from synthetic import make_traces, offset_trace_ids


def test_full_pipeline(tmp_path: Path, xgboost_available):
    frame = make_traces(num_sites=5, train_per_site=3, test_per_site=2)
    background = offset_trace_ids(make_traces(num_sites=0, train_per_site=0, test_per_site=0, background=4))
    input_length = 8
    encoder = DFMiniEncoder(input_length=input_length)

    sequences = list(
        iter_flow_sequences(
            frame, split="train", input_length=input_length,
            min_payload_packets=2, skip_filtered=False,
        )
    )
    embeddings = embed_sequences(sequences, encoder)
    vocabulary, metadata = build_atom_vocabulary(
        embeddings,
        tmp_path / "vocab",
        clusters=5,
        minimum_cluster_size=1,
        scaling=False,
        xgb_sample_size=len(embeddings),
        xgboost_params={"n_estimators": 5, "max_depth": 2},
        seed=0,
    )
    assert metadata["effective_atoms"] <= 5

    extraction = ExtractionConfig(input_length=input_length, batch_size=4, minimum_nonzero_payload_packets=2)
    table = extract_trace_atoms(frame, encoder, vocabulary, extraction)
    background_table = extract_trace_atoms(background, encoder, vocabulary, extraction)

    specs = build_closed_world_specs(frame, seed=0, train_samples=40, validation_samples=20)
    validate_specs(specs, frame)
    result = train_window_predictor(
        specs,
        table,
        tmp_path / "run",
        WindowPredictorConfig(
            seed=0, epochs=3, minimum_epochs=1, patience=2, hidden_dim=8, batch_size=8
        ),
    )
    assert 0.0 <= result["test"]["overall"]["micro_f1"] <= 1.0

    open_specs = build_open_world_specs(
        frame,
        background["trace_id"].astype(int).tolist(),
        specs["labels"],
        seed=0,
        samples_per_cell=3,
        monitored_counts=(1, 2),
        background_counts=(0, 2),
    )
    combined = concatenate_trace_atom_tables([table, background_table])
    predictor = load_frozen_predictor(tmp_path / "run")
    opened = evaluate_frozen(predictor, open_specs, combined, mode="open_world")
    assert 0.0 <= opened["overall"]["micro_f1"] <= 1.0
    assert opened["protocol"]["target_used_for_training"] is False
    saved = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert saved["protocol"]["selection_split"] == "validation"
