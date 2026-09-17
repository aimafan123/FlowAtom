"""The DF-mini convolutional flow encoder.

The architecture follows the lightweight convolutional encoder used by the
Deep Fingerprinting family of website-fingerprinting attacks. It maps a
fixed-length packet-length sequence to a flat flow embedding.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class DFMiniEncoder(nn.Module):
    """Encode a packet-length sequence into a flat flow embedding.

    Parameters
    ----------
    input_length:
        Number of packet positions in the input sequence.
    input_channels:
        ``1`` for a single signed or unsigned length channel. Two channels can
        be used to feed absolute length and direction separately, but such a
        model must be trained from scratch.
    dropout:
        Dropout probability applied after every pooling block.
    """

    def __init__(
        self,
        input_length: int = 300,
        input_channels: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_length <= 0:
            raise ValueError("input_length must be positive")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")

        self.input_length = int(input_length)
        self.input_channels = int(input_channels)
        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(input_channels, 16, kernel_size=8, stride=1, padding="same"),
                nn.BatchNorm1d(16),
                nn.ELU(alpha=1.0),
                nn.Conv1d(16, 16, kernel_size=8, stride=1, padding="same"),
                nn.BatchNorm1d(16),
                nn.ELU(alpha=1.0),
                nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
                nn.Dropout(dropout),
            ),
            nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=8, stride=1, padding="same"),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=8, stride=1, padding="same"),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
                nn.Dropout(dropout),
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=8, stride=1, padding="same"),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=8, stride=1, padding="same"),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
                nn.Dropout(dropout),
            ),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, self.input_length)
            self.output_dim = int(self.encode(dummy).shape[1])

    def _as_channel_first(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            if self.input_channels != 1:
                raise ValueError("2-D input is only valid for a one-channel encoder")
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3:
            raise ValueError(
                "DFMiniEncoder expects [batch, length] or [batch, channels, length]"
            )
        actual: Tuple[int, int] = (int(inputs.shape[1]), int(inputs.shape[2]))
        expected = (self.input_channels, self.input_length)
        if actual != expected:
            raise ValueError(
                f"unexpected flow input shape {actual}, expected {expected}"
            )
        return inputs

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the flattened convolutional flow embedding."""

        return torch.flatten(self.features(self._as_channel_first(inputs)), start_dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encode(inputs)
