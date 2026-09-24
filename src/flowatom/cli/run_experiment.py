"""Run one scenario and one seed using a frozen encoder."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from flowatom import __version__
from flowatom.artifacts import file_sha256, fingerprint
from flowatom.config import DEFAULT_CONFIG_PATH, load_config


def build_plan(traces, encoder, output, config, seed, device, background=None):
    """Describe the dependency-ordered stages with explicit stochastic seeds."""

    def command(name, *args):
        return [sys.executable, "-m", "flowatom", name, *map(str, args)]

    specs = output / "closed_world_specs.json"
    atoms = output / "trace_atoms.npz"
    vocab = output / "vocabulary"
    predictor = output / "predictor"
    common = ["--config", config]
    steps = [
        (
            "split",
            command(
                "build-mixture-specs",
                "--traces",
                traces,
                "--output",
                specs,
                *common,
                "--seed",
                seed,
            ),
        ),
        (
            "vocabulary",
            command(
                "build-atom-vocabulary",
                "--traces",
                traces,
                "--specs",
                specs,
                "--encoder",
                encoder,
                "--output-dir",
                vocab,
                *common,
                "--seed",
                seed,
                "--device",
                device,
            ),
        ),
        (
            "trace_atoms",
            command(
                "build-trace-atoms",
                "--traces",
                traces,
                "--encoder",
                encoder,
                "--vocabulary",
                vocab,
                "--output",
                atoms,
                *common,
                "--flow-sampling-seed",
                seed,
                "--device",
                device,
            ),
        ),
        (
            "train",
            command(
                "train-window-predictor",
                "--specs",
                specs,
                "--atoms",
                atoms,
                "--output-dir",
                predictor,
                *common,
                "--seed",
                seed,
                "--device",
                device,
            ),
        ),
    ]
    if background is not None:
        bg_atoms = output / "background_atoms.npz"
        opened = output / "open_world_specs.json"
        steps.extend(
            [
                (
                    "background_atoms",
                    command(
                        "build-trace-atoms",
                        "--traces",
                        background,
                        "--encoder",
                        encoder,
                        "--vocabulary",
                        vocab,
                        "--output",
                        bg_atoms,
                        *common,
                        "--flow-sampling-seed",
                        seed,
                        "--device",
                        device,
                    ),
                ),
                (
                    "open_specs",
                    command(
                        "build-open-world-specs",
                        "--monitored-traces",
                        traces,
                        "--background-traces",
                        background,
                        "--source-specs",
                        specs,
                        "--output",
                        opened,
                        *common,
                        "--seed",
                        seed,
                    ),
                ),
                (
                    "open_evaluate",
                    command(
                        "evaluate-frozen",
                        "--source-run",
                        predictor,
                        "--evaluation-specs",
                        opened,
                        "--atoms",
                        atoms,
                        "--atoms",
                        bg_atoms,
                        "--output",
                        output / "open_world_metrics.json",
                        "--mode",
                        "open_world",
                        "--device",
                        device,
                    ),
                ),
            ]
        )
    return steps


def validate_background(traces, background):
    """Reject monitored sites/IDs masquerading as background before training."""
    import pandas as pd

    columns = ["trace_id", "label", "site"]
    monitored = pd.read_parquet(traces, columns=columns)
    other = pd.read_parquet(background, columns=columns)
    if other.empty or not (other["label"] == -1).all():
        raise ValueError("background traces must be non-empty and all have label -1")
    if set(monitored["trace_id"]) & set(other["trace_id"]):
        raise ValueError("monitored and background trace IDs overlap")
    if set(monitored["site"]) & set(other["site"]):
        raise ValueError("monitored and background sites overlap")


def save_state(path, state):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path, help="Existing frozen checkpoint.")
    parser.add_argument("--background-traces", type=Path, default=None)
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New or empty run directory."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without creating files."
    )
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    inputs = {"traces": args.traces, "encoder": args.encoder}
    if args.background_traces is not None:
        inputs["background"] = args.background_traces
    inputs = {name: path.expanduser().resolve() for name, path in inputs.items()}
    config_path = args.config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path in [*inputs.values(), config_path]:
        if not path.is_file():
            parser.error(f"input file not found: {path}")
    config = load_config(config_path)
    config["atom_vocabulary"]["seed"] = args.seed
    config["predictor"]["seeds"] = [args.seed]
    config.pop("config_path", None)
    snapshot = output / "config.yaml"
    plan = build_plan(
        inputs["traces"],
        inputs["encoder"],
        output,
        snapshot,
        args.seed,
        args.device,
        inputs.get("background"),
    )
    if args.dry_run:
        print(json.dumps({"seed": args.seed, "stages": dict(plan)}, indent=2))
        return 0
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output directory must be empty: {output}; choose a new run directory")
    if "background" in inputs:
        try:
            validate_background(inputs["traces"], inputs["background"])
        except ValueError as error:
            parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    try:
        with manifest.open("x") as handle:
            handle.write("{}\n")
    except FileExistsError:
        parser.error(f"another run already owns {output}")
    state = {
        "schema_version": 1,
        "status": "running",
        "started_at": time.time(),
        "seed": args.seed,
        "device_requested": args.device,
        "encoder_reused": True,
        "selection_split": "validation",
        "stages": [],
    }
    save_state(manifest, state)
    try:
        import yaml

        snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
        state["inputs"] = {
            name: {"path": str(path), "sha256": file_sha256(path)} for name, path in inputs.items()
        }
        state["config_sha256"] = file_sha256(snapshot)
        state["source_config"] = {"path": str(config_path), "sha256": file_sha256(config_path)}
        package = Path(__file__).resolve().parents[1]
        source_files = {
            str(path.relative_to(package)): file_sha256(path)
            for path in sorted(package.rglob("*.py"))
        }
        state["code_sha256"] = fingerprint(source_files)
        versions = {}
        for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "torch", "xgboost"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        state["environment"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "flowatom": __version__,
            "packages": versions,
        }
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(package.parent), environment.get("PYTHONPATH")])
        )
        (output / "logs").mkdir()
        save_state(manifest, state)
        for name, command in plan:
            stage = {
                "name": name,
                "command": command,
                "status": "running",
                "started_at": time.time(),
                "log": f"logs/{name}.log",
            }
            state["stages"].append(stage)
            save_state(manifest, state)
            print(f"[{len(state['stages'])}/{len(plan)}] {name}", flush=True)
            with (output / stage["log"]).open("w") as log:
                process = subprocess.run(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            stage["returncode"] = process.returncode
            stage["seconds"] = time.time() - stage["started_at"]
            stage["status"] = "completed" if process.returncode == 0 else "failed"
            save_state(manifest, state)
            if process.returncode:
                raise RuntimeError(f"{name} failed; see {output / stage['log']}")
        state["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error"] = str(error) or type(error).__name__
        if state["stages"] and state["stages"][-1]["status"] == "running":
            state["stages"][-1]["status"] = state["status"]
        print(f"FlowAtom run {state['status']}: {state['error']}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        state["finished_at"] = time.time()
        save_state(manifest, state)
    print(f"RUN COMPLETED: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
