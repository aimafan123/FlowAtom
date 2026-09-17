"""Pretraining loop for the FlowAtom MoCo encoder."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from flowatom.pretraining.moco import MoCo


@dataclass(frozen=True)
class MoCoConfig:
    """Hyperparameters for contrastive flow-representation pretraining."""

    epochs: int = 45
    batch_size: int = 8192
    learning_rate: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 0.0001
    schedule: Tuple[int, ...] = (60, 80)
    feature_dim: int = 128
    queue_size: int = 65536
    moco_momentum: float = 0.999
    temperature: float = 0.07
    workers: int = 16
    print_frequency: int = 10
    checkpoint_every: int = 5
    seed: int = 2025
    device: str = "auto"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def train_moco(
    dataset,
    output_dir: Path,
    config: MoCoConfig = MoCoConfig(),
    *,
    resume: Optional[Path] = None,
) -> dict:
    """Train MoCo and return the run history.

    Every ``checkpoint_every`` epochs a ``checkpoint_epoch{N}.pth.tar`` file is
    written and the final momentum encoder is stored as ``encoder.pth.tar``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _resolve_device(config.device)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=config.workers > 0,
    )
    if len(loader) == 0:
        raise ValueError(
            f"dataset rows ({len(dataset)}) must be at least batch_size "
            f"({config.batch_size})"
        )

    model = MoCo(
        input_length=dataset.input_length,
        feature_dim=config.feature_dim,
        queue_size=config.queue_size,
        momentum=config.moco_momentum,
        temperature=config.temperature,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    history = []
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))

    status_path = output_dir / "status.json"
    for epoch in range(start_epoch, config.epochs):
        learning_rate = config.learning_rate
        for milestone in config.schedule:
            if epoch >= milestone:
                learning_rate *= 0.1
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        loss_sum = accuracy_sum = samples = 0
        started = time.time()
        for batch, (query, key) in enumerate(loader):
            query = query.to(device, non_blocking=True)
            key = key.to(device, non_blocking=True)
            logits, labels = model(query, key)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            size = int(query.shape[0])
            loss_sum += float(loss.item()) * size
            accuracy_sum += int((logits.argmax(1) == labels).sum().item())
            samples += size
            if batch % config.print_frequency == 0:
                _atomic_json(
                    status_path,
                    {
                        "status": "training",
                        "epoch": epoch,
                        "batch": batch,
                        "batches": len(loader),
                        "loss": loss_sum / samples,
                        "accuracy": accuracy_sum / samples,
                    },
                )
        summary = {
            "epoch": epoch,
            "loss": loss_sum / samples,
            "accuracy": accuracy_sum / samples,
            "seconds": time.time() - started,
            "learning_rate": learning_rate,
        }
        history.append(summary)
        print(json.dumps(summary), flush=True)
        if epoch % config.checkpoint_every == 0 or epoch == config.epochs - 1:
            torch.save(
                model.checkpoint_payload(
                    epoch=epoch + 1,
                    optimizer=optimizer.state_dict(),
                    history=history,
                ),
                output_dir / f"checkpoint_epoch{epoch}.pth.tar",
            )

    torch.save(
        model.checkpoint_payload(epoch=config.epochs, history=history),
        output_dir / "encoder.pth.tar",
    )
    _atomic_json(
        status_path,
        {"status": "done", "epoch": config.epochs, "config": asdict(config)},
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"history": history, "checkpoint": str(output_dir / "encoder.pth.tar")}
