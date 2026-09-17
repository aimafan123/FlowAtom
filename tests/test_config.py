"""Tests for configuration loading."""

from pathlib import Path

import pytest

from flowatom.config import DEFAULT_CONFIG, ConfigError, load_config, section

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mainline_config_loads_paper_values():
    config = load_config(REPO_ROOT / "configs" / "mainline.yaml")
    encoder = section(config, "encoder")
    assert encoder["input_length"] == 300
    assert encoder["minimum_nonzero_payload_packets"] == 10
    vocabulary = section(config, "atom_vocabulary")
    assert vocabulary["clusters"] == 1200
    assert vocabulary["embedding_scaling"] is False
    predictor = section(config, "predictor")
    assert predictor["seeds"] == [2025, 2026, 2027, 2028, 2029]
    assert config["window"]["max_websites_per_window"] == 5


def test_partial_config_merges_with_defaults(tmp_path: Path):
    path = tmp_path / "partial.yaml"
    path.write_text("atom_vocabulary:\n  clusters: 7\n")
    config = load_config(path)
    assert config["atom_vocabulary"]["clusters"] == 7
    assert config["atom_vocabulary"]["minimum_cluster_size"] == DEFAULT_CONFIG["atom_vocabulary"]["minimum_cluster_size"]
    assert config["encoder"]["input_length"] == 300


def test_missing_config_raises(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_smoke_config_is_smaller_than_mainline():
    smoke = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    mainline = load_config(REPO_ROOT / "configs" / "mainline.yaml")
    assert smoke["pretraining"]["epochs"] < mainline["pretraining"]["epochs"]
    assert smoke["atom_vocabulary"]["clusters"] < mainline["atom_vocabulary"]["clusters"]
    assert smoke["predictor"]["epochs"] < mainline["predictor"]["epochs"]
