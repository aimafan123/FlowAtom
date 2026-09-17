"""Checkpoint loading for the DF-mini flow encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import torch

from flowatom.models.encoder import DFMiniEncoder

PathLike = Union[str, Path]

_MOCO_PREFIXES = (
    "module.encoder_k.",
    "encoder_k.",
    "module.encoder_q.",
    "encoder_q.",
)
_LEGACY_BLOCK_PREFIXES = (
    ("feature_layers.", "features."),
    ("block1.", "features.0."),
    ("block2.", "features.1."),
    ("block3.", "features.2."),
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint does not match the encoder architecture."""


def _load_torch(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        return torch.load(path, map_location=device)


def _strip_moco_prefix(state: Mapping[str, Any]) -> Dict[str, Any]:
    for prefix in _MOCO_PREFIXES:
        selected = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        if selected:
            return selected
    return dict(state)


def _remap_feature_keys(state: Mapping[str, Any]) -> Dict[str, Any]:
    remapped: Dict[str, Any] = {}
    for key, value in state.items():
        for legacy, current in _LEGACY_BLOCK_PREFIXES:
            if key.startswith(legacy):
                key = current + key[len(legacy) :]
                break
        if key.startswith("features."):
            remapped[key] = value
    return remapped


def load_pretrained_encoder(
    checkpoint_path: PathLike,
    input_length: int,
    device: Union[str, torch.device] = "cpu",
) -> DFMiniEncoder:
    """Load a frozen DF-mini encoder from a pretraining or saved checkpoint.

    Both MoCo checkpoints (with a ``state_dict`` member containing
    ``encoder_k``/``encoder_q`` parameters) and plain feature-layer state
    dictionaries are supported. Only convolutional feature layers are read;
    projection and classification heads are ignored.
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device(device)
    payload = _load_torch(checkpoint_path, device)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state = _strip_moco_prefix(payload["state_dict"])
    elif isinstance(payload, Mapping):
        state = dict(payload)
    else:
        raise CheckpointError(
            f"unsupported checkpoint payload: {type(payload).__name__}"
        )

    features = _remap_feature_keys(state)
    if not features:
        raise CheckpointError(
            f"checkpoint {checkpoint_path} contains no DF-mini feature layers"
        )
    model = DFMiniEncoder(input_length=input_length)
    incompatible = model.load_state_dict(features, strict=False)
    missing = [key for key in incompatible.missing_keys if key.startswith("features.")]
    unexpected = [
        key for key in incompatible.unexpected_keys if key.startswith("features.")
    ]
    if missing or unexpected:
        raise CheckpointError(
            f"incompatible encoder checkpoint {checkpoint_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model


def encoder_state_dict(model: DFMiniEncoder) -> Dict[str, Any]:
    """Return the feature-layer state dictionary of an encoder."""

    return {key: value for key, value in model.state_dict().items() if key.startswith("features.")}
