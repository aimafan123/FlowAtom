"""Memory-mapped pretraining shards and the contrastive dataset."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from flowatom.pretraining.augmentation import AUGMENTATION_POLICIES, CopyRTO, TCPViewAugmenter

PathLike = Union[str, Path]
PRETRAINING_COLUMNS = ("signed_length", "payload_length", "direction")


def _fixed_sequence_matrix(chunk) -> np.ndarray:
    """Return a ``[rows, length]`` matrix from a fixed-length list column."""

    rows = len(chunk)
    if rows == 0:
        return np.zeros((0, 0), dtype=np.int64)
    values = np.asarray(chunk.values)
    total = len(values)
    if total % rows != 0:
        raise ValueError("pretraining sequences must all have the same length")
    length = total // rows
    offsets = getattr(chunk, "offsets", None)
    if offsets is not None:
        offsets = np.asarray(offsets)
        if len(offsets) == rows + 1 and not np.all(np.diff(offsets) == length):
            raise ValueError("pretraining sequences must all have the same length")
    return values.reshape(rows, length)


def count_parquet_rows(path: PathLike) -> int:
    """Return the number of rows in a pretraining parquet file."""

    import pyarrow.parquet as pq

    return int(pq.ParquetFile(Path(path)).metadata.num_rows)


def iter_parquet_sequences(
    path: PathLike,
    *,
    column: str = "signed_length",
    input_length: Optional[int] = None,
    max_rows: int = 0,
    seed: int = 2025,
):
    """Yield fixed-length signed sequences from a pretraining parquet file.

    When ``max_rows`` is set, rows are sampled uniformly without replacement
    with the given ``seed`` before being yielded.
    """

    import pyarrow.parquet as pq

    if column not in PRETRAINING_COLUMNS:
        raise ValueError(f"unsupported pretraining column: {column}")
    parquet = pq.ParquetFile(Path(path))
    total_rows = int(parquet.metadata.num_rows)
    selected = None
    if max_rows:
        if max_rows <= 0 or max_rows > total_rows:
            raise ValueError("max_rows must be between 1 and the parquet row count")
        selected = np.sort(
            np.random.default_rng(seed).choice(total_rows, size=max_rows, replace=False)
        )
    offset = 0
    for row_group in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(row_group, columns=[column]).column(0).combine_chunks()
        values = _fixed_sequence_matrix(chunk)
        size = int(values.shape[1])
        if input_length is not None and size != int(input_length):
            raise ValueError(
                f"pretraining sequence length {size} != input_length {input_length}"
            )
        if selected is not None:
            stop = offset + len(values)
            local = selected[(selected >= offset) & (selected < stop)] - offset
            offset = stop
            values = values[local]
        for row in values:
            yield np.asarray(row, dtype=np.int32)



def materialize_shards(
    input_parquet: PathLike,
    output_dir: PathLike,
    *,
    column: str = "signed_length",
    rows: int = 0,
    seed: int = 2025,
) -> dict:
    """Convert a fixed-sequence parquet column into mmap-friendly NPY shards.

    The output directory receives one ``.npy`` file per parquet row group plus
    a ``manifest.json`` consumed by :class:`ShardedSequenceDataset`.
    """

    import pyarrow.parquet as pq

    if column not in PRETRAINING_COLUMNS:
        raise ValueError(f"unsupported pretraining column: {column}")
    input_parquet = Path(input_parquet)
    output_dir = Path(output_dir)
    parquet = pq.ParquetFile(input_parquet)
    total_rows = int(parquet.metadata.num_rows)
    selected = None
    if rows:
        if rows <= 0 or rows > total_rows:
            raise ValueError("rows must be between 1 and the parquet row count")
        selected = np.sort(
            np.random.default_rng(seed).choice(total_rows, size=rows, replace=False)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: List[dict] = []
    input_length: Optional[int] = None
    offset = 0
    for row_group in range(parquet.num_row_groups):
        column_data = parquet.read_row_group(row_group, columns=[column]).column(0)
        chunk = column_data.combine_chunks()
        values = _fixed_sequence_matrix(chunk)
        size = int(values.shape[1])
        values = values.astype(np.int32, copy=False)
        if selected is not None:
            stop = offset + len(values)
            local = selected[(selected >= offset) & (selected < stop)] - offset
            offset = stop
            values = values[local]
            if not len(values):
                continue
        input_length = size
        name = f"{column}_{row_group:04d}.npy"
        np.save(output_dir / name, values)
        shards.append({"path": name, "rows": int(len(values))})
    manifest = {
        "schema_version": 1,
        "source": str(input_parquet),
        "column": column,
        "sampling_seed": seed if selected is not None else None,
        "input_length": input_length,
        "rows": sum(item["rows"] for item in shards),
        "shards": shards,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class ShardedSequenceDataset(Dataset):
    """Serve ``(view_1, view_2)`` pairs from memory-mapped NPY shards.

    ``identity`` and ``copyrto`` return the original sequence as the first view
    and an augmented (or identical) copy as the second view. ``four-way`` and
    ``rosetta-like`` draw two independent augmented views.
    """

    def __init__(
        self,
        manifest: PathLike,
        augment: bool = True,
        augmentation: Optional[str] = None,
    ) -> None:
        manifest = Path(manifest)
        record = json.loads(manifest.read_text())
        self.root = manifest.parent
        self.paths = [self.root / item["path"] for item in record["shards"]]
        self.lengths = [int(item["rows"]) for item in record["shards"]]
        if not self.lengths:
            raise ValueError("pretraining manifest contains no shards")
        self.ends = np.cumsum(self.lengths).tolist()
        self.input_length = int(record["input_length"])
        policy = augmentation or ("copyrto" if augment else "identity")
        if policy not in AUGMENTATION_POLICIES:
            raise ValueError(f"unsupported augmentation policy: {policy}")
        if not augment and policy != "identity":
            raise ValueError("augment=False is only compatible with identity")
        self.augmentation = policy
        if policy == "copyrto":
            self.augmenter = CopyRTO()
        elif policy in {"four-way", "rosetta-like"}:
            self.augmenter = TCPViewAugmenter(policy)
        else:
            self.augmenter = None
        self._arrays: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return int(self.ends[-1]) if self.ends else 0

    def _array(self, shard: int) -> np.ndarray:
        if shard not in self._arrays:
            self._arrays[shard] = np.load(self.paths[shard], mmap_mode="r")
        return self._arrays[shard]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.ends, index)
        start = 0 if shard == 0 else self.ends[shard - 1]
        sequence = torch.tensor(
            np.array(self._array(shard)[index - start], copy=True),
            dtype=torch.float32,
        )
        if self.augmentation in {"four-way", "rosetta-like"}:
            return self.augmenter(sequence), self.augmenter(sequence)
        augmented = self.augmenter(sequence) if self.augmenter else sequence.clone()
        return sequence, augmented
