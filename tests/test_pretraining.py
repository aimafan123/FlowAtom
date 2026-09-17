"""Tests for augmentations, pretraining shards and MoCo."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from flowatom.pretraining import (
    CopyFR,
    CopyRTO,
    MoCo,
    PacketSizeVariation,
    ShardedSequenceDataset,
    ShiftFR,
    ShiftRTO,
    TCPViewAugmenter,
    count_parquet_rows,
    iter_parquet_sequences,
    materialize_shards,
)


@pytest.mark.parametrize(
    "transform",
    [CopyRTO(), CopyFR(), ShiftRTO(), ShiftFR(), PacketSizeVariation()],
)
def test_augmentations_preserve_length_and_dtype(transform):
    sequence = torch.tensor([100.0, -50.0, 300.0, -20.0, 0.0, 0.0], dtype=torch.float32)
    output = transform(sequence)
    assert output.shape == sequence.shape
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("policy", ["four-way", "rosetta-like"])
def test_paired_view_augmenter_preserves_length(policy):
    augmenter = TCPViewAugmenter(policy)
    sequence = torch.tensor([100.0, -50.0, 300.0, -20.0, 40.0, 0.0], dtype=torch.float32)
    view = augmenter(sequence)
    assert view.shape == sequence.shape


def _write_pretraining_parquet(path: Path, rows: int = 10, length: int = 6) -> Path:
    rng = np.random.default_rng(0)
    values = rng.integers(-1500, 1500, size=(rows, length)).astype(np.int64)
    frame = pd.DataFrame({"signed_length": [row.tolist() for row in values]})
    frame.to_parquet(path, index=False)
    return path


def test_materialize_shards_and_dataset(tmp_path: Path):
    parquet = _write_pretraining_parquet(tmp_path / "pretraining.parquet")
    manifest = materialize_shards(parquet, tmp_path / "shards")
    assert manifest["rows"] == 10
    assert manifest["input_length"] == 6
    dataset = ShardedSequenceDataset(tmp_path / "shards" / "manifest.json", augment=False)
    assert len(dataset) == 10
    original, augmented = dataset[0]
    assert original.shape == (6,)
    assert torch.equal(original, augmented)
    augmented_dataset = ShardedSequenceDataset(
        tmp_path / "shards" / "manifest.json", augmentation="copyrto"
    )
    assert augmented_dataset[0][1].shape == (6,)


def test_iter_parquet_sequences_and_subsample(tmp_path: Path):
    parquet = _write_pretraining_parquet(tmp_path / "pretraining.parquet")
    assert count_parquet_rows(parquet) == 10
    assert len(list(iter_parquet_sequences(parquet, input_length=6))) == 10
    assert len(list(iter_parquet_sequences(parquet, input_length=6, max_rows=4, seed=1))) == 4
    with pytest.raises(ValueError):
        list(iter_parquet_sequences(parquet, input_length=5))


def test_moco_forward_advances_queue_and_momentum():
    torch.manual_seed(0)
    model = MoCo(input_length=8, feature_dim=4, queue_size=8, momentum=0.5, temperature=0.1)
    query_parameter = next(iter(model.encoder_q.parameters()))
    key_parameter = next(iter(model.encoder_k.parameters()))
    with torch.no_grad():
        query_parameter.add_(1.0)
    key_before = key_parameter.clone()
    logits, labels = model(torch.randn(4, 8), torch.randn(4, 8))
    assert logits.shape == (4, 9)
    assert labels.shape == (4,)
    assert int(model.queue_ptr.item()) == 4
    # The key encoder moved halfway toward the perturbed query encoder.
    expected = key_before * 0.5 + query_parameter * 0.5
    assert torch.allclose(key_parameter, expected)
    assert all(not parameter.requires_grad for parameter in model.encoder_k.parameters())


def test_moco_requires_divisible_queue():
    model = MoCo(input_length=8, feature_dim=4, queue_size=6)
    with pytest.raises(ValueError):
        model(torch.randn(4, 8), torch.randn(4, 8))


def test_moco_checkpoint_payload_contains_encoder_k():
    model = MoCo(input_length=8, feature_dim=4, queue_size=8)
    payload = model.checkpoint_payload(epoch=3)
    assert payload["epoch"] == 3
    assert any(key.startswith("encoder_k.") for key in payload["state_dict"])
