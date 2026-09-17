"""Momentum Contrast (MoCo) pretraining for flow representations."""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
from torch import nn

from flowatom.models import DFMiniEncoder


class MoCo(nn.Module):
    """MoCo with a queue of negative flow embeddings.

    ``encoder_q`` is trained by gradient descent while ``encoder_k`` is an
    exponential-moving-average copy used to produce stable positive keys. After
    pretraining, ``encoder_k`` is the frozen flow encoder used by FlowAtom.
    """

    def __init__(
        self,
        *,
        input_length: int = 300,
        feature_dim: int = 128,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or queue_size <= 0:
            raise ValueError("feature_dim and queue_size must be positive")
        self.input_length = int(input_length)
        self.feature_dim = int(feature_dim)
        self.queue_size = int(queue_size)
        self.momentum = float(momentum)
        self.temperature = float(temperature)

        self.encoder_q = DFMiniEncoder(input_length=input_length)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        hidden = self.encoder_q.output_dim
        self.projector_q = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.feature_dim),
        )
        self.projector_k = copy.deepcopy(self.projector_q)
        for parameter in self.encoder_k.parameters():
            parameter.requires_grad = False
        for parameter in self.projector_k.parameters():
            parameter.requires_grad = False

        queue = nn.functional.normalize(torch.randn(self.feature_dim, self.queue_size), dim=0)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _key_modules(self):
        yield from self.encoder_k.parameters()
        yield from self.projector_k.parameters()

    def _query_modules(self):
        yield from self.encoder_q.parameters()
        yield from self.projector_q.parameters()

    @torch.no_grad()
    def update_key_encoder(self) -> None:
        for query, key in zip(self._query_modules(), self._key_modules()):
            key.data.mul_(self.momentum).add_(query.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        batch_size = int(keys.shape[0])
        if self.queue_size % batch_size != 0:
            raise ValueError("queue_size must be divisible by batch_size")
        pointer = int(self.queue_ptr.item())
        self.queue[:, pointer : pointer + batch_size] = keys.T
        self.queue_ptr[0] = (pointer + batch_size) % self.queue_size

    def forward(
        self, query_view: torch.Tensor, key_view: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = nn.functional.normalize(self.projector_q(self.encoder_q(query_view)), dim=1)
        with torch.no_grad():
            self.update_key_encoder()
            key = nn.functional.normalize(
                self.projector_k(self.encoder_k(key_view)), dim=1
            )
        positive = torch.einsum("nc,nc->n", query, key).unsqueeze(1)
        negative = torch.einsum("nc,ck->nk", query, self.queue.detach().clone())
        logits = torch.cat((positive, negative), dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self.enqueue(key)
        return logits, labels

    @torch.no_grad()
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return frozen momentum-encoder embeddings for downstream use."""

        return self.encoder_k.encode(inputs)

    def checkpoint_payload(self, **extra) -> Dict:
        """Return a serializable MoCo checkpoint with a ``state_dict`` member."""

        payload = {
            "arch": "DFMiniEncoder",
            "input_length": self.input_length,
            "feature_dim": self.feature_dim,
            "queue_size": self.queue_size,
            "momentum": self.momentum,
            "temperature": self.temperature,
            "state_dict": self.state_dict(),
        }
        payload.update(extra)
        return payload
