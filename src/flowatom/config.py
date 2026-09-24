"""Configuration loading for FlowAtom experiments.

Configurations are plain nested dictionaries loaded from YAML. A small set of
built-in defaults keeps every key well-defined when a user supplies a partial
configuration file.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Optional, Union

PathLike = Union[str, Path]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "mainline.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "encoder": {
        "representation": "signed_payload_length",
        "input_length": 300,
        "minimum_nonzero_payload_packets": 10,
    },
    "pretraining": {
        "feature_dim": 128,
        "queue_size": 65536,
        "momentum": 0.999,
        "temperature": 0.07,
        "epochs": 45,
        "batch_size": 8192,
        "learning_rate": 0.03,
        "momentum_sgd": 0.9,
        "weight_decay": 0.0001,
        "schedule": [60, 80],
        "augmentation": "copyrto",
        "workers": 16,
        "seed": 2025,
    },
    "atom_vocabulary": {
        "clusters": 1200,
        "minimum_cluster_size": 50,
        "embedding_scaling": False,
        "kmeans_n_init": 3,
        "kmeans_batch_size": 8192,
        "xgb_sample_size": 100000,
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "seed": 2025,
    },
    "window": {
        "confidence_threshold": 0.0,
        "max_websites_per_window": 5,
    },
    "predictor": {
        "hidden_dim": 512,
        "dropout": 0.2,
        "epochs": 60,
        "patience": 10,
        "batch_size": 512,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "seeds": [2025, 2026, 2027, 2028, 2029],
    },
    "closed_world": {
        "train_samples": 15000,
        "validation_samples": 1500,
        "test_samples": {1: 200, 2: 100, 3: 80, 4: 60, 5: 40},
    },
    "open_world": {
        "samples_per_cell": 500,
        "monitored_counts": [1, 2, 3, 4, 5],
        "background_counts": [1, 2, 3, 4, 5],
    },
}


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Optional[PathLike] = None) -> dict[str, Any]:
    """Return the default configuration merged with an optional YAML file."""

    import yaml

    config_path = Path(path).expanduser() if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, Mapping):
        raise ConfigError(f"configuration must be a mapping: {config_path}")
    config = _deep_merge(DEFAULT_CONFIG, payload)
    config["config_path"] = str(config_path)
    return config


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return one required configuration section as a plain dictionary."""

    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"configuration section is missing or invalid: {name}")
    return dict(value)
