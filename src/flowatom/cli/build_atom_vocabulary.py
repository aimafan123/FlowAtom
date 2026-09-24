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

import numpy as np
import torch

from flowatom.atoms import build_atom_vocabulary
from flowatom.artifacts import (
    file_sha256, fingerprint, load_embedding_cache, write_embedding_metadata, require_equal,
)
from flowatom.config import load_config, section
from flowatom.data import (
    iter_flow_sequences, load_trace_frame, load_specs, validate_specs,
    validate_trace_pools, validate_trace_frame,
)
from flowatom.encoding import iter_embedding_batches
from flowatom.models import load_pretrained_encoder
from flowatom.pretraining import count_parquet_rows, iter_parquet_sequences


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--traces", type=Path, help="Target-protocol trace parquet.")
    source.add_argument("--pretraining-parquet", type=Path, help="Independent unlabeled parquet.")
    parser.add_argument("--specs", type=Path, help="Closed-world split specs; required with --traces.")
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
    args = parser.parse_args(argv)
    if args.traces is not None and args.specs is None:
        parser.error("--traces requires --specs; build split specs before the Atom vocabulary")
    if args.pretraining_parquet is not None and args.specs is not None:
        parser.error("--specs is only used with --traces")
    if args.max_flows < 0:
        parser.error("--max-flows must be non-negative")

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

    source_sha256 = file_sha256(args.traces or args.pretraining_parquet)
    encoder = load_pretrained_encoder(args.encoder, input_length, device)
    provenance = {
        "encoder_sha256": encoder.checkpoint_sha256,
        "representation": representation,
        "input_length": input_length,
        "minimum_nonzero_payload_packets": min_packets,
    }
    if args.traces is not None:
        frame = load_trace_frame(args.traces)
        validate_trace_frame(frame)
        specs = load_specs(args.specs)
        require_equal(specs.get("trace_dataset_sha256"), source_sha256, "specs trace dataset")
        pools = validate_trace_pools(specs)
        validate_specs(specs, frame)
        frame = frame[frame["trace_id"].isin(pools["train"])].sort_values("trace_id")
        def training_sequences():
            return iter_flow_sequences(
                frame, representation=representation, input_length=input_length,
                min_payload_packets=min_packets, skip_filtered=True,
            )
        # Count after filtering so mmap allocation exactly matches emitted flows.
        flow_count = sum(1 for _ in training_sequences())
        if args.max_flows:
            flow_count = min(flow_count, args.max_flows)
        sequences = islice(training_sequences(), flow_count)
        provenance["atom_fit"] = {
            "source": "trace_train_pool",
            "trace_pools_sha256": fingerprint(specs["trace_pools"]),
            "training_trace_ids": sorted(pools["train"]),
        }
        source_description = {
            "training_source_type": "downstream_trace_train_pool",
            "traces": str(args.traces), "specs": str(args.specs),
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
        expected_column = {"signed_payload_length": "signed_length",
                           "payload_length": "payload_length", "direction_only": "direction"}
        if args.column != expected_column.get(representation):
            parser.error(f"{representation} requires --column {expected_column.get(representation)}")
        provenance["atom_fit"] = {"source": "independent_unlabeled_pretraining"}
        source_description = {
            "training_source_type": "independent_unlabeled_pretraining",
            "pretraining_parquet": str(args.pretraining_parquet),
            "pretraining_column": args.column,
        }

    provenance["source_sha256"] = source_sha256
    cache_contract = {**provenance, "max_flows": args.max_flows, "column": args.column, "seed": seed}
    if flow_count == 0:
        raise ValueError("no eligible training flows remain after short-flow filtering")
    shape = (flow_count, encoder.output_dim)
    if embedding_cache.is_file() and not args.rebuild_embeddings:
        embeddings = load_embedding_cache(embedding_cache, cache_contract, shape)
        print(f"reusing embeddings {embedding_cache} shape={embeddings.shape}", flush=True)
    else:
        embedding_cache.parent.mkdir(parents=True, exist_ok=True)
        # Never expose a partial array as a reusable completed cache.
        temporary_cache = Path(str(embedding_cache) + ".tmp")
        output = np.lib.format.open_memmap(
            temporary_cache, mode="w+", dtype=np.float32, shape=shape
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
        temporary_cache.replace(embedding_cache)
        write_embedding_metadata(embedding_cache, cache_contract, shape)
        embeddings = load_embedding_cache(embedding_cache, cache_contract, shape)

    vocabulary, metadata = build_atom_vocabulary(
        embeddings,
        args.output_dir,
        provenance=provenance,
        clusters=clusters,
        minimum_cluster_size=int(atom_config["minimum_cluster_size"]),
        scaling=scaling,
        kmeans_n_init=int(atom_config["kmeans_n_init"]),
        kmeans_batch_size=int(atom_config["kmeans_batch_size"]),
        xgb_sample_size=int(atom_config["xgb_sample_size"]),
        xgboost_params=dict(atom_config["xgboost"]),
        seed=seed,
        tree_device=str(device),
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
        Path(str(embedding_cache) + ".json").unlink(missing_ok=True)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
