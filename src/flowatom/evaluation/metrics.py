"""Window-level multi-label metrics and threshold decoding.

The primary metric is Micro-F1: true positives, false positives and false
negatives are pooled over all windows before precision and recall are computed.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def window_metrics(predictions: Sequence[Iterable[int]], truths: Sequence[Iterable[int]]) -> Dict[str, float]:
    """Set-based multi-label metrics pooled over windows (Micro-F1 in ``f1``)."""

    if len(predictions) != len(truths):
        raise ValueError("predictions and truths must align")
    if not predictions:
        raise ValueError("cannot evaluate an empty prediction collection")
    pred_sets = [set(int(value) for value in row) for row in predictions]
    truth_sets = [set(int(value) for value in row) for row in truths]
    tp = sum(len(pred & truth) for pred, truth in zip(pred_sets, truth_sets))
    fp = sum(len(pred - truth) for pred, truth in zip(pred_sets, truth_sets))
    fn = sum(len(truth - pred) for pred, truth in zip(pred_sets, truth_sets))
    denominator = sum(
        max(len(pred), len(truth)) for pred, truth in zip(pred_sets, truth_sets)
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    unions = [pred | truth for pred, truth in zip(pred_sets, truth_sets)]
    intersections = [pred & truth for pred, truth in zip(pred_sets, truth_sets)]
    cardinality_error = [len(pred) - len(truth) for pred, truth in zip(pred_sets, truth_sets)]
    result = {
        "micro_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "precision": precision,
        "recall": recall,
        "accuracy": tp / denominator if denominator else 0.0,
        "average_predictions": float(np.mean([len(pred) for pred in pred_sets])),
        "samples": len(pred_sets),
        "mean_jaccard": float(
            np.mean(
                [
                    len(intersection) / len(union) if union else 1.0
                    for intersection, union in zip(intersections, unions)
                ]
            )
        ),
        "exact_match": float(
            np.mean([pred == truth for pred, truth in zip(pred_sets, truth_sets)])
        ),
        "fp_per_window": float(np.mean([len(pred - truth) for pred, truth in zip(pred_sets, truth_sets)])),
        "cardinality_mae": float(np.mean([abs(value) for value in cardinality_error])),
        "cardinality_bias": float(np.mean(cardinality_error)),
    }
    if all(not truth for truth in truth_sets):
        result.update(
            {
                "window_false_positive_rate": float(
                    np.mean([bool(pred) for pred in pred_sets])
                ),
                "exact_rejection_rate": float(
                    np.mean([not pred for pred in pred_sets])
                ),
            }
        )
    return result


def decode_threshold(
    probabilities: np.ndarray,
    labels: Sequence[int],
    threshold: float,
    max_websites: int = 5,
) -> List[List[int]]:
    """Decode a website set per window by thresholding site probabilities."""

    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [windows, sites]")
    predictions = []
    for row in probabilities:
        selected = np.flatnonzero(row >= float(threshold))
        if len(selected) > max_websites:
            selected = selected[np.argsort(row[selected])[-max_websites:]]
        predictions.append([int(labels[index]) for index in selected])
    return predictions


def evaluate_by_size(predictions: Sequence[Iterable[int]], specs: Sequence[Mapping]) -> Dict[str, Dict[str, float]]:
    """Report metrics for each true website-set size plus the overall pool."""

    output = {}
    for websites_num in range(1, 6):
        indices = [
            index
            for index, spec in enumerate(specs)
            if int(spec["websites_num"]) == websites_num
        ]
        if not indices:
            continue
        output[str(websites_num)] = window_metrics(
            [predictions[index] for index in indices],
            [specs[index]["labels"] for index in indices],
        )
    output["overall"] = window_metrics(
        predictions, [spec["labels"] for spec in specs]
    )
    return output


def group_metrics(
    predictions: Sequence[Iterable[int]],
    specs: Sequence[Mapping],
    fields: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Group windows by integer metadata fields and score every group."""

    if len(predictions) != len(specs):
        raise ValueError("predictions and specs must align")
    output: Dict[str, Dict[str, float]] = {}
    keys = sorted(
        {tuple(int(spec[field]) for field in fields) for spec in specs if all(field in spec for field in fields)}
    )
    for key in keys:
        indices = [
            index
            for index, spec in enumerate(specs)
            if tuple(int(spec[field]) for field in fields) == key
        ]
        output["/".join(map(str, key))] = window_metrics(
            [predictions[index] for index in indices],
            [specs[index]["labels"] for index in indices],
        )
    return output


def evaluate_open_world(
    predictions: Sequence[Iterable[int]], specs: Sequence[Mapping]
) -> Dict[str, Dict[str, float]]:
    """Report open-world metrics overall and per (monitored, background) cell."""

    fields = ["monitored_websites_num", "background_traces_num"]
    if not all(all(field in spec for field in fields) for spec in specs):
        fields = ["monitored_websites_num", "background_websites_num"]
    output = {"overall": window_metrics(predictions, [spec["labels"] for spec in specs])}
    output.update(group_metrics(predictions, specs, fields))
    for field in fields:
        for name, metrics in group_metrics(predictions, specs, [field]).items():
            output[f"{field}={name}"] = metrics
    return output


def mean_micro_f1(metrics_per_seed: Sequence[Mapping[str, Mapping[str, float]]], group: str = "overall") -> Dict[str, float]:
    """Mean and sample standard deviation of Micro-F1 across seeds."""

    values = [float(metrics[group]["micro_f1"]) for metrics in metrics_per_seed]
    if not values:
        raise ValueError("at least one seed result is required")
    return {
        "mean_micro_f1": float(np.mean(values)),
        "std_micro_f1": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "seeds": len(values),
    }
