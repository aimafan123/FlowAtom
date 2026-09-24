"""Public command-line entry points, imported lazily for lightweight help."""

import argparse
import importlib
import sys

from flowatom import __version__

COMMANDS = {
    "build-atom-vocabulary": "build_atom_vocabulary",
    "build-mixture-specs": "build_mixture_specs",
    "build-open-world-specs": "build_open_world_specs",
    "build-trace-atoms": "build_trace_atoms",
    "evaluate-frozen": "evaluate_frozen",
    "make-synthetic-data": "make_synthetic_data",
    "materialize-pretraining-shards": "materialize_pretraining_shards",
    "pretrain-encoder": "pretrain_encoder",
    "summarize-results": "summarize_results",
    "train-window-predictor": "train_window_predictor",
    "run": "run_experiment",
}


def main(argv=None):
    """Dispatch a subcommand while preserving its own argument parser."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="flowatom", description="FlowAtom research pipeline")
    parser.add_argument("--version", action="version", version=f"flowatom {__version__}")
    parser.add_argument("command", choices=sorted(COMMANDS))
    if not arguments or arguments[0] in ("-h", "--help", "--version"):
        parser.parse_args(arguments or ["--help"])
    selected = parser.parse_args(arguments[:1]).command
    module = importlib.import_module(f"flowatom.cli.{COMMANDS[selected]}")
    return module.main(arguments[1:])
