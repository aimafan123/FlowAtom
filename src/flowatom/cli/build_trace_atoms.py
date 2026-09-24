#!/usr/bin/env python3
"""Extract a per-trace Atom response table from a trace parquet file."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from flowatom.atoms import AtomVocabulary, ExtractionConfig, extract_trace_atoms
from flowatom.config import load_config, section
from flowatom.artifacts import file_sha256
from flowatom.data import load_trace_frame
from flowatom.models import load_pretrained_encoder


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True, help="Atom vocabulary directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--representation", default=None)
    parser.add_argument("--minimum-nonzero-payload-packets", type=int, default=None)
    parser.add_argument("--max-flows-per-trace", type=int, default=0)
    parser.add_argument("--flow-sampling-seed", type=int, default=2025)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    encoder_config = section(config, "encoder")
    window_config = section(config, "window")
    input_length = args.input_length or int(encoder_config["input_length"])
    device = torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    frame = load_trace_frame(args.traces)
    encoder = load_pretrained_encoder(args.encoder, input_length, device)
    vocabulary = AtomVocabulary.load(args.vocabulary)
    table = extract_trace_atoms(
        frame,
        encoder,
        vocabulary,
        ExtractionConfig(
            input_length=input_length,
            batch_size=args.batch_size,
            confidence_threshold=args.threshold
            if args.threshold is not None
            else float(window_config["confidence_threshold"]),
            representation=args.representation or str(encoder_config["representation"]),
            minimum_nonzero_payload_packets=args.minimum_nonzero_payload_packets
            if args.minimum_nonzero_payload_packets is not None
            else int(encoder_config["minimum_nonzero_payload_packets"]),
            max_flows_per_trace=args.max_flows_per_trace,
            flow_sampling_seed=args.flow_sampling_seed,
            device=str(device),
        ),
    )
    table = replace(table, source_sha256=file_sha256(args.traces))
    table.save(args.output)
    summary = {
        "output": str(args.output),
        "traces": table.trace_count,
        "atoms": table.atom_count,
        "flows": table.total_flow_count,
        "retained_flows": table.retained_flow_count,
        "short_flows": table.short_flow_count,
        "zero_response_traces": int((table.flow_count == 0).sum()),
        "confidence_threshold": table.confidence_threshold,
        "device": str(device),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
