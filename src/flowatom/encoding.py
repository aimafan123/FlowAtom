"""Batched flow embedding with a frozen encoder."""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Union

import numpy as np
import torch


def encode_batch(
    sequence_batch: Union[np.ndarray, Sequence[np.ndarray]],
    encoder,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Encode one batch of fixed-length sequences into embeddings."""

    values = np.asarray(sequence_batch, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("sequence batch must be a 2-D array")
    device = torch.device(device)
    inputs = torch.from_numpy(values).to(device)
    with torch.no_grad():
        embeddings = encoder.encode(inputs).cpu().numpy()
    return embeddings.astype(np.float32)


def iter_embedding_batches(
    sequences: Iterable[np.ndarray],
    encoder,
    *,
    batch_size: int = 2048,
    device: Union[str, torch.device] = "cpu",
) -> Iterator[np.ndarray]:
    """Yield embedding matrices for a stream of sequences."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch = []
    for sequence in sequences:
        batch.append(sequence)
        if len(batch) >= batch_size:
            yield encode_batch(batch, encoder, device)
            batch = []
    if batch:
        yield encode_batch(batch, encoder, device)


def embed_sequences(
    sequences: Union[np.ndarray, Sequence[np.ndarray]],
    encoder,
    *,
    batch_size: int = 2048,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Encode all sequences and return a ``[flows, embedding_dim]`` matrix."""

    values = np.asarray(sequences, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("sequences must form a 2-D array")
    batches = [
        batch
        for batch in iter_embedding_batches(values, encoder, batch_size=batch_size, device=device)
    ]
    if not batches:
        raise ValueError("cannot embed an empty sequence collection")
    return np.concatenate(batches, axis=0)
