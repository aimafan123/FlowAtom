"""Train and evaluate the window-level website-set predictor."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flowatom.atoms.table import TraceAtomTable
from flowatom.evaluation import (
    decode_threshold,
    evaluate_by_size,
    evaluate_open_world,
    group_metrics,
    window_metrics,
)
from flowatom.models import WindowSetPredictor
from flowatom.window import FeatureStandardizer, materialize_windows

PathLike = Union[str, Path]
THRESHOLD_GRID = tuple(float(value) for value in np.arange(0.05, 0.951, 0.025))


@dataclass(frozen=True)
class WindowPredictorConfig:
    """Hyperparameters of the window-set decoder."""

    seed: int = 2025
    batch_size: int = 512
    epochs: int = 60
    minimum_epochs: int = 10
    patience: int = 10
    hidden_dim: int = 512
    dropout: float = 0.2
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_websites: int = 5
    device: str = "auto"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def predict_probabilities(
    model: WindowSetPredictor,
    features: np.ndarray,
    device: Union[str, torch.device],
    batch_size: int,
) -> np.ndarray:
    """Return sigmoid site probabilities for a window feature matrix."""

    device = torch.device(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(features, dtype=np.float32))),
        batch_size=batch_size,
        shuffle=False,
    )
    outputs = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            outputs.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def select_threshold(
    probabilities: np.ndarray,
    truths: Sequence[Sequence[int]],
    labels: Sequence[int],
    max_websites: int = 5,
) -> Tuple[float, float, Dict[str, float]]:
    """Select the decoding threshold that maximizes validation Micro-F1."""

    best_f1, best_threshold, best_metrics = -1.0, THRESHOLD_GRID[0], {}
    for threshold in THRESHOLD_GRID:
        predictions = decode_threshold(probabilities, labels, threshold, max_websites)
        metrics = window_metrics(predictions, truths)
        if metrics["micro_f1"] > best_f1:
            best_f1 = metrics["micro_f1"]
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_f1, best_metrics


def train_window_predictor(
    specs: Mapping,
    atoms: TraceAtomTable,
    output_dir: PathLike,
    config: WindowPredictorConfig = WindowPredictorConfig(),
    *,
    validation_size_filter: Optional[Sequence[int]] = None,
) -> Dict:
    """Train the decoder with validation-only model and threshold selection."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = resolve_device(config.device)

    labels = [int(value) for value in specs["labels"]]
    train_x, train_y = materialize_windows(specs["train"], atoms, labels)
    validation_specs = list(specs["validation"])
    if validation_size_filter is not None:
        allowed = {int(value) for value in validation_size_filter}
        validation_specs = [
            spec for spec in validation_specs if int(spec["websites_num"]) in allowed
        ]
    val_x, _ = materialize_windows(validation_specs, atoms, labels)
    test_x, _ = materialize_windows(specs["test"], atoms, labels)

    standardizer = FeatureStandardizer.fit(train_x)
    train_x = standardizer.transform(train_x)
    val_x = standardizer.transform(val_x)
    test_x = standardizer.transform(test_x)

    model = WindowSetPredictor(
        train_x.shape[1], len(labels), config.hidden_dim, config.dropout
    ).to(device)
    positives = train_y.sum(axis=0)
    positive_weight = torch.from_numpy(
        ((len(train_y) - positives) / np.maximum(positives, 1)).astype(np.float32)
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x.astype(np.float32)),
            torch.from_numpy(train_y.astype(np.float32)),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_truths = [spec["labels"] for spec in validation_specs]
    best: Optional[Dict] = None
    stale = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            loss = criterion(logits, batch_y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item()) * len(batch_x)
        val_probabilities = predict_probabilities(model, val_x, device, config.batch_size)
        threshold, f1, metrics = select_threshold(
            val_probabilities, validation_truths, labels, config.max_websites
        )
        record = {
            "epoch": epoch,
            "loss": total / len(train_x),
            "validation_micro_f1": f1,
            "threshold": threshold,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if best is None or f1 > best["validation_micro_f1"] + 1e-5:
            best = {
                "state": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "validation_micro_f1": f1,
                "threshold": threshold,
                "validation_metrics": metrics,
            }
            stale = 0
        else:
            stale += 1
            if epoch >= config.minimum_epochs and stale >= config.patience:
                break
    assert best is not None
    model.load_state_dict(best["state"])
    test_probabilities = predict_probabilities(model, test_x, device, config.batch_size)
    predictions = decode_threshold(
        test_probabilities, labels, best["threshold"], config.max_websites
    )
    result = {
        "method": "flowatom_max_pooling",
        "seed": config.seed,
        "labels": labels,
        "input_dim": int(train_x.shape[1]),
        "atom_count": int(atoms.atom_count),
        "best_epoch": best["epoch"],
        "threshold": best["threshold"],
        "validation_micro_f1": best["validation_micro_f1"],
        "validation": best["validation_metrics"],
        "test": evaluate_by_size(predictions, specs["test"]),
        "config": asdict(config),
        "protocol": {
            "window_features": "peak (max pooling over flows)",
            "selection_split": "validation",
            "test_used_for_selection": False,
            "website_count_input": False,
        },
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    torch.save(model.state_dict(), output_dir / "model.pt")
    standardizer.save(output_dir / "standardizer.npz")
    return result


@dataclass
class FrozenPredictor:
    """A trained decoder frozen for target-domain or open-world evaluation."""

    model: WindowSetPredictor
    standardizer: FeatureStandardizer
    labels: Sequence[int]
    threshold: float
    max_websites: int
    batch_size: int
    source_run: Path
    metrics: Dict


def load_frozen_predictor(run_dir: PathLike, device: str = "cpu") -> FrozenPredictor:
    """Load a trained decoder and its validation-selected threshold."""

    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    device_object = resolve_device(device)
    labels = [int(value) for value in metrics["labels"]]
    config = metrics["config"]
    model = WindowSetPredictor(
        int(metrics["input_dim"]), len(labels), int(config["hidden_dim"]), float(config["dropout"])
    )
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.to(device_object).eval()
    return FrozenPredictor(
        model=model,
        standardizer=FeatureStandardizer.load(run_dir / "standardizer.npz"),
        labels=labels,
        threshold=float(metrics["threshold"]),
        max_websites=int(config["max_websites"]),
        batch_size=int(config["batch_size"]),
        source_run=run_dir,
        metrics=metrics,
    )


def evaluate_frozen(
    predictor: FrozenPredictor,
    specs: Mapping,
    atoms: TraceAtomTable,
    *,
    mode: str = "closed_world",
    device: str = "cpu",
    max_websites: Optional[int] = None,
) -> Dict:
    """Evaluate a frozen decoder on closed-world, drift or open-world windows."""

    if mode not in {"closed_world", "open_world"}:
        raise ValueError(f"unsupported evaluation mode: {mode}")
    device_object = resolve_device(device)
    predictor.model.to(device_object)
    limit = int(max_websites) if max_websites is not None else predictor.max_websites
    features, _ = materialize_windows(specs["test"], atoms, predictor.labels)
    features = predictor.standardizer.transform(features)
    probabilities = predict_probabilities(
        predictor.model, features, device_object, predictor.batch_size
    )
    predictions = decode_threshold(probabilities, predictor.labels, predictor.threshold, limit)
    if mode == "open_world":
        grouped = evaluate_open_world(predictions, specs["test"])
    else:
        grouped = evaluate_by_size(predictions, specs["test"])
    return {
        "method": "flowatom_max_pooling_frozen",
        "mode": mode,
        "source_run": str(predictor.source_run),
        "frozen_threshold": float(predictor.threshold),
        "max_websites": limit,
        "overall": window_metrics(predictions, [spec["labels"] for spec in specs["test"]]),
        "groups": grouped,
        "protocol": {
            "source_only_training": True,
            "target_used_for_training": False,
            "target_used_for_selection": False,
            "decoder_frozen_from_source_validation": True,
        },
    }


def evaluate_specs_grouped(
    predictor: FrozenPredictor,
    specs: Mapping,
    atoms: TraceAtomTable,
    fields: Sequence[str],
    *,
    device: str = "cpu",
    max_websites: Optional[int] = None,
) -> Dict:
    """Evaluate a frozen decoder and group windows by arbitrary metadata fields."""

    limit = int(max_websites) if max_websites is not None else predictor.max_websites
    device_object = resolve_device(device)
    predictor.model.to(device_object)
    features, _ = materialize_windows(specs["test"], atoms, predictor.labels)
    features = predictor.standardizer.transform(features)
    probabilities = predict_probabilities(
        predictor.model, features, device_object, predictor.batch_size
    )
    predictions = decode_threshold(probabilities, predictor.labels, predictor.threshold, limit)
    return {
        "overall": window_metrics(predictions, [spec["labels"] for spec in specs["test"]]),
        "groups": group_metrics(predictions, specs["test"], fields),
    }
