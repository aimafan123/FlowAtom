"""Tests for the DF-mini encoder and checkpoint loading."""

from pathlib import Path

import pytest
import torch

from flowatom.models import (
    CheckpointError,
    DFMiniEncoder,
    encoder_state_dict,
    load_pretrained_encoder,
)


def test_output_dimension_matches_paper_encoder():
    encoder = DFMiniEncoder(input_length=300)
    assert encoder.output_dim == 2496
    assert encoder.encode(torch.zeros(2, 300)).shape == (2, 2496)


def test_encode_accepts_two_dimensional_input():
    encoder = DFMiniEncoder(input_length=16)
    encoder.eval()
    outputs = encoder.encode(torch.zeros(3, 16))
    assert outputs.shape[0] == 3


def test_shape_mismatch_raises():
    encoder = DFMiniEncoder(input_length=16)
    with pytest.raises(ValueError):
        encoder.encode(torch.zeros(2, 17))


def test_two_channel_encoder_rejects_two_dimensional_input():
    encoder = DFMiniEncoder(input_length=16, input_channels=2)
    with pytest.raises(ValueError):
        encoder.encode(torch.zeros(2, 16))
    assert encoder.encode(torch.zeros(2, 2, 16)).shape[0] == 2


def test_moco_style_checkpoint_round_trip(tmp_path: Path):
    encoder = DFMiniEncoder(input_length=16)
    payload = {
        "state_dict": {
            f"module.encoder_k.{key}": value for key, value in encoder.state_dict().items()
        }
    }
    path = tmp_path / "moco.pth.tar"
    torch.save(payload, path)
    loaded = load_pretrained_encoder(path, 16)
    inputs = torch.randn(3, 16)
    assert torch.equal(encoder.eval().encode(inputs), loaded.encode(inputs))


def test_plain_feature_state_dict_round_trip(tmp_path: Path):
    encoder = DFMiniEncoder(input_length=16)
    path = tmp_path / "encoder.pt"
    torch.save(encoder_state_dict(encoder), path)
    loaded = load_pretrained_encoder(path, 16)
    inputs = torch.randn(2, 16)
    assert torch.equal(encoder.eval().encode(inputs), loaded.encode(inputs))


def test_legacy_block_keys_are_remapped(tmp_path: Path):
    encoder = DFMiniEncoder(input_length=16)
    state = {}
    for key, value in encoder.state_dict().items():
        for index, legacy in enumerate(("block1.", "block2.", "block3.")):
            if key.startswith(f"features.{index}."):
                state[legacy + key[len(f"features.{index}.") :]] = value
    path = tmp_path / "legacy.pt"
    torch.save(state, path)
    loaded = load_pretrained_encoder(path, 16)
    inputs = torch.randn(2, 16)
    assert torch.equal(encoder.eval().encode(inputs), loaded.encode(inputs))


def test_missing_feature_layers_raise(tmp_path: Path):
    path = tmp_path / "empty.pt"
    torch.save({"fc_out.weight": torch.zeros(1, 1)}, path)
    with pytest.raises(CheckpointError):
        load_pretrained_encoder(path, 16)
