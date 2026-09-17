# FlowAtom

**Atom-based evidence aggregation for multi-label website fingerprinting.**

FlowAtom identifies the *set* of websites present in a window of encrypted
traffic. A single flow usually carries only partial evidence of website
identity, so FlowAtom learns shared local traffic prototypes called **Atoms**
without website labels, represents every flow by its responses to those Atoms,
and max-pools the responses across all flows in an observation window into a
fixed-dimensional, permutation-invariant representation. A small multi-label
MLP then predicts the website set.

This repository is a clean, self-contained release of the code used for the
paper. It reproduces the method end to end and includes a synthetic dataset so
the full pipeline can be exercised without the original capture data.

## Repository layout

```
configs/            Paper and smoke-test configurations
docs/               Method, data-format and reproduction documentation
scripts/            Command-line entry points for every pipeline stage
src/flowatom/       Reusable implementation
  representation.py   Signed payload-length flow representation
  models/             DF-mini flow encoder, checkpoint loading, MLP decoder
  pretraining/        TCP-aware augmentations, sharded dataset, MoCo trainer
  atoms/              Atom vocabulary construction and trace-response extraction
  window/             Permutation-invariant window representation
  data/               Trace schema, window specifications and validation
  evaluation/         Micro-F1 metrics and threshold decoding
  training/           Window-set predictor training and frozen evaluation
tests/              Unit, parity and end-to-end tests
```

## Method in one page

1. **Flow representation pretraining.** Zero-payload packets are removed and a
   flow becomes a signed payload-length sequence
   `abs(packet length) * sign(direction)`, truncated or zero-padded to
   `L = 300`. A DF-mini convolutional encoder is trained on large-scale
   unlabeled traffic with MoCo-style contrastive learning, using TCP-aware
   augmentations from the multi-tab WF literature.
2. **Atom construction.** The frozen encoder embeds the training flows of the
   target traffic scenario. k-means clusters these embeddings; clusters smaller
   than a minimum size are discarded and the retained centers are the **Atoms**.
   An XGBoost mapper is trained on the cluster assignments as pseudo-labels and
   maps a flow embedding to a soft Atom response vector. No website label is
   used in this stage.
3. **Window-set prediction.** For each window, every flow is encoded and mapped
   to Atom responses, and a per-Atom max pooling over flows produces the window
   representation `Φ_a(T) = max_{f∈T} q_{f,a}`. Standardization parameters are
   fitted on training windows only. A two-hidden-layer MLP is trained with
   positive-weighted binary cross-entropy, and the decoding threshold is
   selected on the validation split. Visit boundaries, flow-to-website
   assignments and the true website count are never provided to the model.

See [`docs/method.md`](docs/method.md) for the full description and
[`docs/reproduction.md`](docs/reproduction.md) for paper-level commands.

## Installation

The code targets Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # or: pip install -e ".[all]"
```

`torch` is used for the encoder and decoder, `xgboost` for the Atom mapper, and
`scikit-learn` for k-means and standardization. Everything can run on a CPU;
a GPU only speeds up pretraining and embedding extraction.

## Quick start: synthetic end-to-end smoke test

The smoke test generates a small synthetic dataset, pretrains a tiny encoder,
builds an Atom vocabulary, extracts trace Atom responses, trains the window
predictor and runs an open-world evaluation:

```bash
PYTHON=.venv/bin/python bash scripts/run_smoke_test.sh
```

Artifacts are written to `artifacts/smoke/`. The run takes well under a minute
on a CPU and prints `SMOKE TEST PASSED` at the end. The synthetic numbers are
only a plumbing check; use the real datasets for meaningful accuracy.

Run the unit and integration tests with:

```bash
PYTHONPATH=src pytest -q
```

## Reproducing the paper

The full pipeline on real data is:

```bash
# 1. Pretraining shards and MoCo encoder (external unlabeled traffic).
python scripts/materialize_pretraining_shards.py \
  --input data/raw/pretraining/jp_mawi/signed_length_L300.parquet \
  --output-dir cache/shards
python scripts/pretrain_encoder.py \
  --manifest cache/shards/manifest.json --output-dir checkpoints/encoder \
  --config configs/mainline.yaml --workers 16

# 2. Scenario-specific Atom vocabulary from the training split (no labels).
python scripts/build_atom_vocabulary.py \
  --traces data/processed/closed_world/direct/traces.parquet \
  --encoder checkpoints/encoder/encoder.pth.tar \
  --output-dir checkpoints/atom_vocabulary/direct --config configs/mainline.yaml

# 3. Trace Atom responses and closed-world window specifications.
python scripts/build_trace_atoms.py \
  --traces data/processed/closed_world/direct/traces.parquet \
  --encoder checkpoints/encoder/encoder.pth.tar \
  --vocabulary checkpoints/atom_vocabulary/direct \
  --output cache/direct/trace_atoms.npz --config configs/mainline.yaml
python scripts/build_mixture_specs.py \
  --traces data/processed/closed_world/direct/traces.parquet \
  --output specs/direct/closed_world.json --config configs/mainline.yaml

# 4. Window-set predictor. The paper reports five seeds.
for seed in 2025 2026 2027 2028 2029; do
  python scripts/train_window_predictor.py \
    --specs specs/direct/closed_world.json \
    --atoms cache/direct/trace_atoms.npz \
    --output-dir runs/direct/seed${seed} --config configs/mainline.yaml --seed ${seed}
done

# 5. Open-world evaluation with unmonitored background traffic.
python scripts/build_trace_atoms.py \
  --traces data/processed/open_world/background/traces.parquet \
  --encoder checkpoints/encoder/encoder.pth.tar \
  --vocabulary checkpoints/atom_vocabulary/direct \
  --output cache/direct/background_trace_atoms.npz --config configs/mainline.yaml
python scripts/build_open_world_specs.py \
  --monitored-traces data/processed/closed_world/direct/traces.parquet \
  --background-traces data/processed/open_world/background/traces.parquet \
  --source-specs specs/direct/closed_world.json \
  --output specs/direct/open_world.json --config configs/mainline.yaml
python scripts/evaluate_frozen.py \
  --source-run runs/direct/seed2025 \
  --evaluation-specs specs/direct/open_world.json \
  --atoms cache/direct/trace_atoms.npz \
  --atoms cache/direct/background_trace_atoms.npz \
  --output runs/direct/seed2025/open_world.json --mode open_world
```

Repeat step 2–4 for the `trojan` and `vmess` scenarios. The Atom vocabulary is
built per traffic scenario, while the pretrained encoder is shared.

## Data

The capture datasets are not distributed with this repository. See
[`docs/data_format.md`](docs/data_format.md) for the exact parquet schemas and
[`docs/reproduction.md`](docs/reproduction.md) for how to point the scripts at
your own captures. `scripts/make_synthetic_data.py` generates data in the same
schema for development.

## What is intentionally not included

This release contains the FlowAtom method and the scripts required to
reproduce it. It does not include the baseline implementations (ARES, BAPM,
TMWF, Flow-DF, CAWF), the large proprietary traffic captures, trained
checkpoints, the cluster/MLflow/DVC experiment orchestration, or exploratory
ablations that are not part of the paper. The comparison baselines are
third-party methods and should be obtained from their own publications.

## Implementation notes

- All deterministic components were checked for numerical parity against the
  original research implementation: flow representations, the DF-mini encoder
  output, window max pooling, Micro-F1 metrics and threshold decoding, and the
  closed-world and open-world window generators produce identical results.
- Atom construction supports memory-mapped embedding caches so that one
  million pretraining flows can be clustered without materializing a second
  full matrix in RAM.
- Target and background traffic are never used for training or for threshold
  selection; this invariant is recorded in every result file.

## Citation

```bibtex
@inproceedings{fan2026flowatom,
  title     = {FlowAtom: Atom-Based Evidence Aggregation for Multi-Label Website Fingerprinting},
  author    = {Fan, Chongru and Huang, Wentao and Wang, Wei and Ding, Zhenquan and Shi, Jinqiao and Cai, Wei and Hao, Zhiyu},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026}
}
```

## License

Released under the Apache License 2.0. See [`LICENSE`](LICENSE).
