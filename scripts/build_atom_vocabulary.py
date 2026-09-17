#!/usr/bin/env python3
"""Build an Atom vocabulary from unlabeled flows.

Flows come either from the training split of the target-protocol traces
(scenario-specific Atoms, the paper mainline) or from an independent unlabeled
pretraining parquet (shared Atoms, an ablation). Website labels are never used.
"""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from flowatom.atoms import build_atom_vocabulary
from flowatom.config import load_config, section
from flowatom.data import iter_flow_sequences, load_trace_frame
from flowatom.encoding import iter_embedding_batches
from flowatom.models import load_pretrained_encoder
from flowatom.pretraining import count_parquet_rows, iter_parquet_sequences


def _count_trace_flows(path: Path) -> int:
    frame = load_trace_frame(path, columns=["trace_id", "split", "payload_flows"])
    frame = frame[frame["split"] == "train"]
    return int(sum(len(flows) for flows in frame["payload_flows"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--traces", type=Path, help="Target-protocol trace parquet.")
    source.add_argument("--pretraining-parquet", type=Path, help="Independent unlabeled parquet.")
    parser.add_argument("--encoder", type=Path, required=True, help="Pretrained encoder checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--scaling", dest="scaling", action="store_true", default=None)
    parser.add_argument("--no-scaling", dest="scaling", action="store_false")
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--representation", default=None)
    parser.add_argument("--minimum-nonzero-payload-packets", type=int, default=None)
    parser.add_argument("--max-flows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--column", choices=("signed_length", "payload_length", "direction"), default="signed_length")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument("--keep-embeddings", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    encoder_config = section(config, "encoder")
    atom_config = section(config, "atom_vocabulary")
    input_length = args.input_length or int(encoder_config["input_length"])
    representation = args.representation or str(encoder_config["representation"])
    min_packets = (
        args.minimum_nonzero_payload_packets
        if args.minimum_nonzero_payload_packets is not None
        else int(encoder_config["minimum_nonzero_payload_packets"])
    )
    clusters = args.clusters if args.clusters is not None else int(atom_config["clusters"])
    scaling = bool(atom_config["embedding_scaling"]) if args.scaling is None else bool(args.scaling)
    seed = args.seed if args.seed is not None else int(atom_config["seed"])
    device = torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache = args.embedding_cache or (args.output_dir / "embeddings.npy")

    if args.traces is not None:
        flow_count = _count_trace_flows(args.traces)
        if args.max_flows:
            flow_count = min(flow_count, args.max_flows)
        frame = load_trace_frame(args.traces)
        sequences = iter_flow_sequences(
            frame,
            split="train",
            representation=representation,
            input_length=input_length,
            min_payload_packets=min_packets,
            skip_filtered=False,
        )
        if args.max_flows:
            sequences = islice(sequences, flow_count)
        source_description = {
            "training_source_type": "downstream_trace_train_split",
            "traces": str(args.traces),
            "max_flows": args.max_flows,
        }
    else:
        available = count_parquet_rows(args.pretraining_parquet)
        flow_count = min(available, args.max_flows) if args.max_flows else available
        sequences = iter_parquet_sequences(
            args.pretraining_parquet,
            column=args.column,
            input_length=input_length,
            max_rows=args.max_flows,
            seed=seed,
        )
        source_description = {
            "training_source_type": "independent_unlabeled_pretraining",
            "pretraining_parquet": str(args.pretraining_parquet),
            "pretraining_column": args.column,
        }

    if embedding_cache.is_file() and not args.rebuild_embeddings:
        embeddings = np.load(embedding_cache, mmap_mode="r")
        print(f"reusing embeddings {embedding_cache} shape={embeddings.shape}", flush=True)
    else:
        encoder = load_pretrained_encoder(args.encoder, input_length, device)
        embedding_cache.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            embedding_cache, mode="w+", dtype=np.float32, shape=(flow_count, encoder.output_dim)
        )
        offset = 0
        for batch in iter_embedding_batches(sequences, encoder, batch_size=args.batch_size, device=device):
            output[offset : offset + len(batch)] = batch
            offset += len(batch)
            if offset % (args.batch_size * 25) == 0:
                print(f"embeddings={offset}/{flow_count}", flush=True)
        if offset != flow_count:
            raise RuntimeError(f"embedding count mismatch: {offset} != {flow_count}")
        output.flush()
        del output
        embeddings = np.load(embedding_cache, mmap_mode="r")

    vocabulary, metadata = build_atom_vocabulary(
        embeddings,
        args.output_dir,
        clusters=clusters,
        minimum_cluster_size=int(atom_config["minimum_cluster_size"]),
        scaling=scaling,
        kmeans_n_init=int(atom_config["kmeans_n_init"]),
        kmeans_batch_size=int(atom_config["kmeans_batch_size"]),
        xgb_sample_size=int(atom_config["xgb_sample_size"]),
        xgboost_params=dict(atom_config["xgboost"]),
        seed=seed,
        transform_chunk_size=16384,
    )
    metadata.update(source_description)
    metadata.update(
        {
            "encoder": str(args.encoder),
            "representation": representation,
            "input_length": input_length,
            "minimum_nonzero_payload_packets": min_packets,
            "embedding_cache": str(embedding_cache),
            "embedding_shape": [int(value) for value in embeddings.shape],
        }
    )
    (args.output_dir / "build_metadata.json").write_text(json.dumps(metadata, indent=2))
    if not args.keep_embeddings:
        embedding_cache.unlink(missing_ok=True)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
