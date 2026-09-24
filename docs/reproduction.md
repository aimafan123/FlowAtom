# Reproduction guide

Run commands from the repository root after [installation](environment.md).
The installed `flowatom` command also works outside the checkout when input,
configuration and output paths are supplied explicitly.

## 1. Prepare data and an encoder

Convert capture data to the [trace parquet schema](data_format.md). Use globally
unique trace IDs; background sites and IDs must not overlap monitored data.
Neither website boundaries nor true set size are predictor inputs.

If a compatible external encoder is available, pass it directly to the runner.
Record the encoder's pretraining provenance and checksum. Otherwise, pretrain
one on a disjoint external traffic dataset:

```bash
flowatom materialize-pretraining-shards \
  --input data/raw/pretraining/signed_length_L300.parquet \
  --output-dir cache/pretraining_shards
flowatom pretrain-encoder \
  --manifest cache/pretraining_shards/manifest.json \
  --output-dir checkpoints/encoder \
  --config configs/mainline.yaml --seed 2025 --workers 16 --device cuda:0
```

The output checkpoint is `checkpoints/encoder/encoder.pth.tar`. Its input length
and representation must match downstream extraction. The mainline uses signed
payload length, length 300, and at least 10 nonzero payload packets per flow.
The same frozen external encoder may be shared across traffic scenarios;
scenario-specific Atoms and predictors are built independently.

## 2. One scenario, one downstream seed

```bash
flowatom run \
  --traces data/processed/direct/traces.parquet \
  --background-traces data/processed/background/traces.parquet \
  --encoder checkpoints/encoder/encoder.pth.tar \
  --output-dir runs/direct/seed2025 \
  --config configs/mainline.yaml --seed 2025 --device cuda:0
```

Omit the background argument for closed world only. The runner validates
background disjointness before fitting models. `--dry-run` prints commands
without writing files. Each actual run uses a new or empty output directory.
It snapshots the resolved YAML, sets downstream seeds explicitly, invokes the
same stage commands used by the source scripts, and stops at the first error.

```text
runs/direct/seed2025/
  manifest.json              Inputs/code/config fingerprints, environment, stage status
  config.yaml                Resolved configuration used by the stages
  logs/                      One log per stage
  closed_world_specs.json    Trace pools and training/validation/test windows
  vocabulary/                Atom mapper and provenance
  trace_atoms.npz             Monitored trace responses
  predictor/                 Model, standardizer, metrics and artifact manifest
  background_atoms.npz       Present when background was supplied
  open_world_specs.json      Target-present open-world windows
  open_world_metrics.json    Frozen open-world metrics
```

Check `manifest.json.status == "completed"` before using a run. Failure and
interruption states retain logs and partial artifacts. The runner does not
resume or overwrite partial runs automatically. For stage-level work, use
[the CLI reference](../scripts/README.md).

## 3. Repeat complete downstream runs across seeds

This loop repeats splitting, Atom fitting, feature extraction, predictor
training and evaluation for every scenario/seed pair:

```bash
for scenario in direct trojan vmess; do
  for seed in 2025 2026 2027 2028 2029; do
    flowatom run \
      --traces "data/processed/${scenario}/traces.parquet" \
      --background-traces data/processed/background/traces.parquet \
      --encoder checkpoints/encoder/encoder.pth.tar \
      --output-dir "runs/${scenario}/seed${seed}" \
      --config configs/mainline.yaml --seed "$seed" --device cuda:0
  done
  flowatom summarize-results --runs-root "runs/${scenario}" \
    --pattern 'seed*/predictor/metrics.json' \
    --output "runs/${scenario}/closed_summary.json"
  flowatom summarize-results --runs-root "runs/${scenario}" \
    --pattern 'seed*/open_world_metrics.json' \
    --output "runs/${scenario}/open_summary.json"
done
```

The encoder remains fixed in this loop. To include encoder-pretraining
variation, train and pass a seed-specific external encoder for each seed and
record that change explicitly. The loop defines a reproducible experiment for
this revision; it does not establish equivalence to every historical paper run.
Changing only the predictor seed while sharing one vocabulary is a narrower
experiment and should be reported as such.

`summary` uses the sample standard deviation (`ddof=1`). For one run, the
standard deviation is `null` because it cannot be estimated from one sample.
Closed-world metrics contain per-size groups `1` through `5`; use `--group 3`
for a specific window size. Keep scenarios and experimental settings separate
when aggregating runs.

## 4. Open-world evaluation and the paper subset

The mainline open-world grid has monitored set size `m=1..5` and background
trace count `b=1..5`, with 500 windows per cell. It is target-present evaluation,
not a background-only rejection test. All three measured scenarios use Direct
HTTPS background data. Thresholds and maximum decoded set size remain frozen
from the source run.

The paper's `b=5, m=2..5` pooled subset can be evaluated without retraining:

```bash
python - <<'PYCODE'
import json
from pathlib import Path
run = Path("runs/direct/seed2025")
specs = json.loads((run / "open_world_specs.json").read_text())
specs["test"] = [w for w in specs["test"]
                 if w["background_traces_num"] == 5
                 and w["monitored_websites_num"] in (2, 3, 4, 5)]
(run / "open_world_paper_subset.json").write_text(json.dumps(specs, indent=2))
PYCODE
flowatom evaluate-frozen \
  --source-run runs/direct/seed2025/predictor \
  --evaluation-specs runs/direct/seed2025/open_world_paper_subset.json \
  --atoms runs/direct/seed2025/trace_atoms.npz \
  --atoms runs/direct/seed2025/background_atoms.npz \
  --output runs/direct/seed2025/open_world_paper_metrics.json \
  --mode open_world --device cuda:0
```

Read `overall.micro_f1` from the subset result. Do not average the per-cell
F1 values: pooled Micro-F1 combines the underlying predictions across windows.

## 5. Drift and ablations

For drift data, extract trace responses with the **source** encoder and
vocabulary, then call `flowatom evaluate-frozen` with the source predictor and
target window specs. Use `--mode closed_world` for known monitored labels or
`--mode open_world` for target-present windows with background. Target labels
must use the source label mapping. Do not fit another standardizer or select a
new threshold on the target data. Raw target window generation and real drift
captures are not bundled.

For Atom count or other hyperparameters, copy `configs/mainline.yaml`, edit the
relevant field (for example `atom_vocabulary.clusters`), and run a fresh complete
experiment with `--config` pointing to that file. Keep the encoder and dataset
fixed when studying that parameter. The runner stores the resolved configuration.

Flow-budget ablations use the stage commands because the full runner keeps all
eligible flows:

```bash
flowatom build-trace-atoms \
  --traces data/processed/direct/traces.parquet \
  --encoder checkpoints/encoder/encoder.pth.tar \
  --vocabulary runs/direct/seed2025/vocabulary \
  --output cache/direct_k4.npz --config configs/mainline.yaml \
  --max-flows-per-trace 4 --flow-sampling-seed 2025 --device cuda:0
flowatom train-window-predictor \
  --specs runs/direct/seed2025/closed_world_specs.json \
  --atoms cache/direct_k4.npz --output-dir runs/direct_k4/seed2025 \
  --config configs/mainline.yaml --seed 2025 --device cuda:0
```

Sampling takes place after short-flow filtering. Re-extract background caches
with the same budget before open-world evaluation. The feature contract rejects
mixing all-flow and budgeted caches.

For an encoder-source ablation, replace the external encoder with a separately
recorded checkpoint and rebuild the vocabulary and every downstream artifact.
To fit shared Atoms on external pretraining traffic, use
`flowatom build-atom-vocabulary --pretraining-parquet ...` and then the individual
extraction/training commands. Do not present external-pool fitting as the
scenario-training-pool mainline.
