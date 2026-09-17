"""Tests for Atom vocabulary construction and trace-response extraction."""

from pathlib import Path

import numpy as np
import pytest
import torch

from flowatom.atoms import (
    AtomVocabulary,
    ExtractionConfig,
    NearestCentroidClassifier,
    TraceAtomTable,
    build_atom_vocabulary,
    classifier_fit_indices,
    concatenate_trace_atom_tables,
    extract_trace_atoms,
)
from flowatom.encoding import embed_sequences
from flowatom.models import DFMiniEncoder
from flowatom.representation import flow_representation


def test_classifier_fit_indices_covers_every_atom():
    labels = np.asarray([0, 1, 2, 0, 1, 2, -1, -1])
    indices = classifier_fit_indices(labels, 2, seed=0)
    assert set(labels[indices].tolist()) == {0, 1, 2}


def test_build_atom_vocabulary_round_trip(tmp_path: Path, xgboost_available):
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(60, 6)).astype(np.float32)
    vocabulary, metadata = build_atom_vocabulary(
        embeddings,
        tmp_path / "vocab",
        clusters=4,
        minimum_cluster_size=1,
        scaling=False,
        xgb_sample_size=60,
        xgboost_params={"n_estimators": 5, "max_depth": 2},
        seed=0,
    )
    assert metadata["effective_atoms"] <= 4
    probabilities = vocabulary.predict_proba(embeddings[:5])
    assert probabilities.shape == (5, metadata["effective_atoms"])
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    loaded = AtomVocabulary.load(tmp_path / "vocab")
    assert loaded.atom_count == vocabulary.atom_count
    assert loaded.embedding_dimension == 6
    assert np.allclose(loaded.predict_proba(embeddings[:5]), probabilities)
    assert (tmp_path / "vocab" / "clusters.npz").is_file()


def test_build_atom_vocabulary_with_scaling(tmp_path: Path, xgboost_available):
    rng = np.random.default_rng(1)
    embeddings = (rng.normal(size=(40, 5)) * 10).astype(np.float32)
    vocabulary, metadata = build_atom_vocabulary(
        embeddings,
        tmp_path / "vocab",
        clusters=3,
        minimum_cluster_size=1,
        scaling=True,
        xgb_sample_size=40,
        xgboost_params={"n_estimators": 5, "max_depth": 2},
        seed=1,
    )
    assert metadata["scaling_enabled"] is True
    assert (tmp_path / "vocab" / "scaler.joblib").is_file()
    assert not (tmp_path / "vocab" / "normalized_embeddings.npy").exists()


def test_nearest_centroid_vocabulary():
    rng = np.random.default_rng(2)
    centers = rng.normal(size=(3, 4)).astype(np.float32)
    classifier = NearestCentroidClassifier(centers)
    probabilities = classifier.predict_proba(centers[:2])
    assert probabilities.shape == (2, 3)
    assert probabilities[0, 0] == 1.0 and probabilities[1, 1] == 1.0
    vocabulary = AtomVocabulary(classifier=classifier, scaler=None, classifier_type="nearest_centroid")
    assert vocabulary.embedding_dimension == 4
    assert vocabulary.atom_count == 3


def test_extract_trace_atoms_matches_direct_computation(tiny_traces):
    torch.manual_seed(0)
    encoder = DFMiniEncoder(input_length=8)
    rng = np.random.default_rng(3)
    centers = rng.normal(size=(3, encoder.output_dim)).astype(np.float32)
    vocabulary = AtomVocabulary(
        classifier=NearestCentroidClassifier(centers), scaler=None, classifier_type="nearest_centroid"
    )
    frame = tiny_traces
    table = extract_trace_atoms(
        frame,
        encoder,
        vocabulary,
        ExtractionConfig(input_length=8, batch_size=3, minimum_nonzero_payload_packets=2),
    )
    assert table.trace_count == len(frame)
    assert table.atom_count == 3
    # Recompute the expected peak responses row by row.
    for row_index in range(len(frame)):
        flows = frame["payload_flows"].iloc[row_index]
        directions = frame["direction_flows"].iloc[row_index]
        sequences = [
            flow_representation(payload, direction, input_length=8, min_payload_packets=2)
            for payload, direction in zip(flows, directions)
        ]
        expected = vocabulary.predict_proba(embed_sequences(sequences, encoder)).max(axis=0)
        assert np.allclose(table.peak[row_index], expected)
    assert table.flow_count.sum() == sum(len(flows) for flows in frame["payload_flows"])


def test_extract_trace_atoms_subsamples_flows(tiny_traces):
    torch.manual_seed(0)
    encoder = DFMiniEncoder(input_length=8)
    rng = np.random.default_rng(4)
    vocabulary = AtomVocabulary(
        classifier=NearestCentroidClassifier(rng.normal(size=(2, encoder.output_dim)).astype(np.float32)),
        scaler=None,
        classifier_type="nearest_centroid",
    )
    table = extract_trace_atoms(
        tiny_traces,
        encoder,
        vocabulary,
        ExtractionConfig(
            input_length=8, batch_size=2, minimum_nonzero_payload_packets=2,
            max_flows_per_trace=1, flow_sampling_seed=11,
        ),
    )
    assert table.flow_count.sum() == len(tiny_traces)
    assert table.flow_count.max() == 1
    assert table.total_flow_count == len(tiny_traces)


def test_trace_atom_table_save_load_and_concatenate(tmp_path: Path):
    table = TraceAtomTable(
        trace_ids=np.asarray([1, 2, 3]),
        labels=np.asarray([0, 1, 0]),
        peak=np.arange(9, dtype=np.float32).reshape(3, 3),
        flow_count=np.asarray([2, 3, 1], dtype=np.float32),
    )
    path = table.save(tmp_path / "atoms.npz")
    loaded = TraceAtomTable.load(path)
    assert np.array_equal(loaded.peak, table.peak)
    assert loaded.atom_count == 3
    second = TraceAtomTable(
        trace_ids=np.asarray([4, 5]),
        labels=np.asarray([1, 0]),
        peak=np.ones((2, 3), dtype=np.float32),
        flow_count=np.asarray([1, 1], dtype=np.float32),
    )
    combined = concatenate_trace_atom_tables([loaded, second])
    assert combined.trace_count == 5
    assert combined.peak.shape == (5, 3)


def test_trace_atom_table_rejects_duplicate_ids():
    table = TraceAtomTable(
        trace_ids=np.asarray([1, 1]),
        labels=np.asarray([0, 1]),
        peak=np.zeros((2, 2), dtype=np.float32),
        flow_count=np.asarray([1, 1], dtype=np.float32),
    )
    with pytest.raises(ValueError):
        table.validate()
