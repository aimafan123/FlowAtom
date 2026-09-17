"""TCP-aware packet-sequence augmentations for contrastive pretraining.

The transformations follow the augmentation policies used by multi-tab WF
literature: retransmission insertion/shifting (RTO and fast-retransmit) and
packet-size coalescing. Each transform preserves packet order and direction
signs.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

AUGMENTATION_POLICIES = ("identity", "copyrto", "four-way", "rosetta-like")


class CopyRTO:
    """Retransmission augmentation that duplicates a short packet run."""

    def __init__(self, loss_mean: float = 0.15, loss_std: float = 0.1) -> None:
        self.loss_mean = float(loss_mean)
        self.loss_std = float(loss_std)

    def _loss_rate(self) -> float:
        while True:
            value = float(np.random.normal(self.loss_mean, self.loss_std))
            if 0.1 <= value <= 0.3:
                return value

    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        loss_rate = self._loss_rate()
        merged: List[torch.Tensor] = []
        pending: List[torch.Tensor] = []
        index = 0
        while index < len(sequence):
            if random.random() < loss_rate:
                end = min(index + random.randint(1, 3), len(sequence))
                segment = list(sequence[index:end])
                pending.extend(segment)
                merged.extend(segment)
                index = end
            else:
                merged.append(sequence[index])
                if pending:
                    merged.extend(pending)
                    pending.clear()
                index += 1
        merged.extend(pending)
        return torch.stack(merged[: len(sequence)]).to(dtype=torch.float32)


class CopyFR:
    """Fast-retransmit augmentation that duplicates individual packets once."""

    def __init__(self, loss_mean: float = 0.15, loss_std: float = 0.1) -> None:
        self.loss_mean = float(loss_mean)
        self.loss_std = float(loss_std)

    def _loss_rate(self) -> float:
        while True:
            value = float(np.random.normal(self.loss_mean, self.loss_std))
            if 0.1 <= value <= 0.3:
                return value

    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        loss_rate = self._loss_rate()
        pending = [[item, False] for item in sequence]
        merged: List[torch.Tensor] = []
        while pending:
            for index, (item, already_lost) in enumerate(pending):
                merged.append(item)
                if random.random() > loss_rate or already_lost:
                    pending.pop(index)
                    break
                pending[index][1] = True
        return torch.stack(merged[: len(sequence)]).to(dtype=torch.float32)


class ShiftRTO:
    """RTO augmentation that shifts a short packet run without duplication."""

    def __init__(self, loss_mean: float = 0.15, loss_std: float = 0.1) -> None:
        self.loss_mean = float(loss_mean)
        self.loss_std = float(loss_std)

    def _loss_rate(self) -> float:
        while True:
            value = float(np.random.normal(self.loss_mean, self.loss_std))
            if 0.1 <= value <= 0.3:
                return value

    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        loss_rate = self._loss_rate()
        merged: List[torch.Tensor] = []
        pending: List[torch.Tensor] = []
        index = 0
        while index < len(sequence):
            if random.random() < loss_rate:
                end = min(index + random.randint(1, 3), len(sequence))
                pending.extend(sequence[index:end])
                index = end
            else:
                merged.append(sequence[index])
                if pending:
                    merged.extend(pending)
                    pending.clear()
                index += 1
        merged.extend(pending)
        return torch.stack(merged[: len(sequence)]).to(dtype=torch.float32)


class ShiftFR:
    """Fast-retransmit augmentation that shifts packets without duplication."""

    def __init__(self, loss_mean: float = 0.15, loss_std: float = 0.1) -> None:
        self.loss_mean = float(loss_mean)
        self.loss_std = float(loss_std)

    def _loss_rate(self) -> float:
        while True:
            value = float(np.random.normal(self.loss_mean, self.loss_std))
            if 0.1 <= value <= 0.3:
                return value

    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        loss_rate = self._loss_rate()
        pending = [[item, False] for item in sequence]
        merged: List[torch.Tensor] = []
        while pending:
            for index, (item, already_lost) in enumerate(pending):
                if random.random() > loss_rate or already_lost:
                    merged.append(item)
                    pending.pop(index)
                    break
                pending[index][1] = True
        return torch.stack(merged[: len(sequence)]).to(dtype=torch.float32)


class PacketSizeVariation:
    """Approximate RTT/MSS coalescing for signed packet-length sequences."""

    def __init__(
        self,
        *,
        maximum_attach_ratio: float = 0.6,
        minimum_mss: int = 500,
        maximum_mss: int = 1500,
    ) -> None:
        self.maximum_attach_ratio = float(maximum_attach_ratio)
        self.minimum_mss = int(minimum_mss)
        self.maximum_mss = int(maximum_mss)

    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        attach_ratio = random.random() * self.maximum_attach_ratio
        mss = random.randint(self.minimum_mss, self.maximum_mss)
        merged: List[float] = []
        for item in sequence.tolist():
            value = float(item)
            same_direction = merged and (merged[-1] > 0) == (value > 0)
            if not same_direction or random.random() > attach_ratio:
                merged.append(value)
                continue
            sign = 1.0 if value > 0 else -1.0
            combined = abs(merged.pop()) + abs(value)
            while combined > mss:
                merged.append(sign * mss)
                combined -= mss
            if combined:
                merged.append(sign * combined)
        return torch.tensor(merged, dtype=torch.float32)


class TCPViewAugmenter:
    """Generate independent TCP-aware views for paired-view policies."""

    FOUR_WAY = ("copyrto", "copyfr", "shiftrto", "shiftfr")
    ROSETTA_LIKE = FOUR_WAY + ("packet_size_variation",)

    def __init__(self, policy: str) -> None:
        if policy not in {"four-way", "rosetta-like"}:
            raise ValueError(f"unsupported paired-view policy: {policy}")
        self.policy = policy
        self.transforms: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "copyrto": CopyRTO(),
            "copyfr": CopyFR(),
            "shiftrto": ShiftRTO(),
            "shiftfr": ShiftFR(),
            "packet_size_variation": PacketSizeVariation(),
        }

    @staticmethod
    def _unpadded(sequence: torch.Tensor) -> torch.Tensor:
        nonzero = torch.nonzero(sequence, as_tuple=False)
        length = int(nonzero[-1].item()) + 1 if len(nonzero) else 0
        return sequence[:length]

    @staticmethod
    def _fit(sequence: torch.Tensor, length: int) -> torch.Tensor:
        sequence = sequence[:length].to(dtype=torch.float32)
        if len(sequence) < length:
            sequence = torch.cat(
                (sequence, torch.zeros(length - len(sequence), dtype=torch.float32))
            )
        return sequence

    def _apply(self, sequence: torch.Tensor, names: Sequence[str]) -> torch.Tensor:
        target_length = len(sequence)
        output = self._unpadded(sequence)
        if not len(output):
            return sequence.clone().to(dtype=torch.float32)
        for name in names:
            output = self.transforms[name](output)
        return self._fit(output, target_length)

    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.policy == "four-way":
            names = [random.choice(self.FOUR_WAY)]
        else:
            names = [name for name in self.ROSETTA_LIKE if random.random() < 0.5]
            if not names:
                names = [random.choice(self.ROSETTA_LIKE)]
            random.shuffle(names)
        return self._apply(sequence, names)
