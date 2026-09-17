"""Tests for multi-label metrics and decoders."""

import numpy as np
import pytest

from flowatom.evaluation import (
    decode_threshold,
    evaluate_by_size,
    evaluate_open_world,
    window_metrics,
)


def test_micro_f1_pools_over_windows():
    metrics = window_metrics([[0, 1], [2], [3, 4, 5]], [[0, 1], [1], [3, 4]])
    assert metrics["f1"] == metrics["micro_f1"]
    assert metrics["precision"] == pytest.approx(4 / 6)
    assert metrics["recall"] == pytest.approx(4 / 5)
    assert metrics["f1"] == pytest.approx(2 * (4 / 6) * (4 / 5) / ((4 / 6) + (4 / 5)))


def test_window_metrics_reports_rejection_when_no_truth():
    metrics = window_metrics([[], [2]], [[], []])
    assert metrics["window_false_positive_rate"] == pytest.approx(0.5)
    assert metrics["exact_rejection_rate"] == pytest.approx(0.5)


def test_decode_threshold_caps_predicted_websites():
    probabilities = np.asarray([[0.9, 0.8, 0.7, 0.6, 0.55, 0.5]], dtype=np.float32)
    predictions = decode_threshold(probabilities, [0, 1, 2, 3, 4, 5], 0.5, max_websites=2)
    assert [sorted(row) for row in predictions] == [[0, 1]]


def test_evaluate_by_size_groups_and_overall():
    specs = [
        {"websites_num": 1, "labels": [0]},
        {"websites_num": 2, "labels": [1, 2]},
    ]
    result = evaluate_by_size([[0], [1, 2]], specs)
    assert set(result) == {"1", "2", "overall"}
    assert result["overall"]["f1"] == pytest.approx(1.0)


def test_evaluate_open_world_groups_by_cell():
    specs = [
        {"websites_num": 1, "monitored_websites_num": 1, "background_traces_num": 0, "labels": [0]},
        {"websites_num": 2, "monitored_websites_num": 2, "background_traces_num": 2, "labels": [1, 2]},
    ]
    result = evaluate_open_world([[0], [1, 2]], specs)
    assert "overall" in result
    assert "1/0" in result
    assert "2/2" in result
