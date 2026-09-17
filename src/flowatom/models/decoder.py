"""Window-level website-set decoder."""

from __future__ import annotations

import torch
from torch import nn


class WindowSetPredictor(nn.Module):
    """Multi-label MLP decoder over a window representation.

    The network has two hidden layers and one logit per monitored website. It
    receives only the permutation-invariant window representation; neither the
    true website count nor any flow-to-website assignment is used.
    """

    def __init__(
        self,
        input_dim: int,
        num_sites: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_sites <= 0:
            raise ValueError("input_dim and num_sites must be positive")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        hidden = int(hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden // 2, num_sites)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(features))

    @property
    def representation_dim(self) -> int:
        return int(self.backbone[0].in_features)
