# FlowAtom

**Atom-based evidence aggregation for multi-label website fingerprinting.**

FlowAtom predicts the set of websites present in a window of encrypted traffic.
A frozen flow encoder maps signed payload-length sequences to embeddings;
a scenario-specific Atom vocabulary maps those embeddings to soft responses.
Max pooling combines the flow responses into a fixed-size window representation,
and a multi-label MLP predicts the website set.

This repository contains the FlowAtom implementation, synthetic example data
generation, and scripts for training and frozen evaluation.

[Method](docs/method.md) · [Reproduction](docs/reproduction.md) ·
[Measured results](docs/results.md) · [Data format](docs/data_format.md)

## Installation

Use Python 3.9 or later. The CPU reference environment is tested on Linux;
see [environment notes](docs/environment.md) for pinned dependencies and CUDA.
Run these commands from a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
flowatom --help
```

The `ml` extra installs PyTorch, `atoms` installs XGBoost, and `all` installs
both. Core data and metric utilities can be installed with `pip install -e .`.

## Quick start

Run the complete synthetic pipeline, including a small encoder pretraining run:

```bash
PYTHON=.venv/bin/python bash scripts/run_smoke_test.sh
```

Each invocation creates a fresh directory under `artifacts/smoke/` and finishes
with `SMOKE TEST PASSED`. It uses the CPU and usually takes seconds to a few
minutes, depending on hardware. Synthetic scores verify the pipeline only.

## Run one experiment

Given a trace parquet and a compatible pretrained encoder:

```bash
flowatom run \
  --traces data/processed/direct/traces.parquet \
  --background-traces data/processed/background/traces.parquet \
  --encoder checkpoints/encoder/encoder.pth.tar \
  --output-dir runs/direct/seed2025 \
  --config configs/mainline.yaml --seed 2025 --device cuda:0
```

Omit `--background-traces` for a closed-world run, or use `--device cpu` without
a GPU. `--dry-run` prints the stage commands. The output directory must be new
or empty. A failed run preserves its logs and status; use a new directory when
retrying the whole pipeline.

The runner fixes the trace split, builds Atoms from the training pool, extracts
features, trains the predictor, and optionally evaluates open-world windows.
Every downstream stochastic stage receives the same seed. `manifest.json`
records input and code fingerprints, package versions, commands and status;
`config.yaml` is a snapshot of the resolved configuration.

For encoder pretraining, five-seed runs, drift evaluation and ablations, follow
[the reproduction guide](docs/reproduction.md). Individual stages remain
available as `flowatom <command>` and through the existing `scripts/*.py` wrappers.

## Reproduction status

A seed-2025 run has been completed for all three scenarios. The table compares
this single-seed reproduction with the five-seed means reported in the paper's
Introduction. Micro-F1 is shown as a percentage.

| Scenario | Closed world: this release / paper | Open world: this release / paper |
| --- | ---: | ---: |
| Direct HTTPS | 97.86 / 97.82 | 92.52 / 92.37 |
| Trojan | 94.31 / 94.43 | 93.34 / 92.64 |
| VMess | 94.28 / 93.92 | 89.60 / 89.85 |

[Results and limitations](docs/results.md) also include scores on the default
generated windows, results for the paper's open-world subset, configuration
and provenance.

## Repository layout

```text
src/flowatom/       Reusable implementation and installed CLI
configs/            Mainline and synthetic-run configurations
docs/               Method, data contracts, environment and reproduction
scripts/            Backward-compatible wrappers and smoke test
tests/              Regression tests for contracts and release workflows
requirements/       Pinned CPU reference dependencies
.github/            Continuous integration and issue/PR templates
```

Data, caches, checkpoints and run outputs are ignored by Git. The default
configurations are also packaged under `src/flowatom/configs/`; tests check that
these copies match `configs/`.

## Tests

```bash
python -m pip install -e ".[all,dev]"
python -m unittest discover -s tests -v
ruff check src scripts tests
```

CI checks regression tests, the synthetic pipeline, source/wheel builds and an
installed wheel outside the checkout. Model/cache fingerprints reject mixed
or stale artifacts. See [migration notes](docs/data_format.md) before loading
artifacts created by an older revision.

## License

The code is released under [MIT](LICENSE). Capture datasets and external
checkpoints have their own distribution conditions; they are not covered by
this code release. Baseline implementations and private cluster orchestration
are outside the repository's scope.
