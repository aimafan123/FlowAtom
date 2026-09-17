# Reproducing the paper

This guide maps every paper experiment to commands in this repository. All
commands assume the repository root as the working directory and
`PYTHONPATH=src` (the scripts bootstrap it automatically; installing the
package with `pip install -e .` is also fine).

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is recommended for pretraining and embedding extraction. Pass
`--device cuda:0` where supported, or use the default `auto`.

## 2. Data used in the paper

| Role | Source | Size |
| --- | --- | --- |
| Pretraining (unlabeled) | JP-MAWI traffic | 1,000,000 flows |
| Monitored websites | Tranco Top 10K | 100 websites |
| Traffic scenarios | Direct HTTPS, Trojan, VMess | per-scenario visit traces |
| Background (open world) | Unmonitored websites from the same list | 1 visit trace each |
| Spatial drift | Direct captures from `au, de, jp, sg, za` | inference only |
| Temporal drift | Direct captures from five monthly dates | inference only |

Pretraining and monitored traffic are disjoint. Monitored websites are
excluded from the open-world background pool.

Convert your captures into the trace parquet described in
[`data_format.md`](data_format.md). The essential requirements are:

- one row per visit trace with a globally unique `trace_id`;
- `split ∈ {train, test}` with disjoint trace sets;
- nested `payload_flows` and `direction_flows` with aligned packet lengths;
- no IP addresses, DNS names or SNI in the model input;
- `label = -1` and `site = "background-*"` for unmonitored traces.

The paper splits single-website visit traces into train, validation and test
trace sets, then builds windows offline. `scripts/build_mixture_specs.py`
implements exactly this split: 90% of each website's training traces form the
training pool, the remaining 10% form the validation pool, and test traces are
disjoint from both.

## 3. Pretrain the flow encoder

```bash
python scripts/materialize_pretraining_shards.py \
  --input data/raw/pretraining/jp_mawi/signed_length_L300.parquet \
  --output-dir cache/pretraining_shards

python scripts/pretrain_encoder.py \
  --manifest cache/pretraining_shards/manifest.json \
  --output-dir checkpoints/encoder \
  --config configs/mainline.yaml \
  --workers 16 --device cuda:0
```

The frozen encoder is `checkpoints/encoder/encoder.pth.tar`. It is shared
across the three traffic scenarios.

## 4. Closed-world experiments (Table 1)

For each scenario `direct`, `trojan`, `vmess`:

```bash
SCENARIO=direct        # or trojan / vmess
TRACES=data/processed/closed_world/${SCENARIO}/traces.parquet

# 4.1 Scenario-specific Atoms from the unlabeled training split.
python scripts/build_atom_vocabulary.py \
  --traces "$TRACES" --encoder checkpoints/encoder/encoder.pth.tar \
  --output-dir checkpoints/atom_vocabulary/${SCENARIO} \
  --config configs/mainline.yaml --device cuda:0

# 4.2 Per-trace Atom responses.
python scripts/build_trace_atoms.py \
  --traces "$TRACES" --encoder checkpoints/encoder/encoder.pth.tar \
  --vocabulary checkpoints/atom_vocabulary/${SCENARIO} \
  --output cache/${SCENARIO}/trace_atoms.npz \
  --config configs/mainline.yaml --device cuda:0

# 4.3 Windows and five training seeds.
python scripts/build_mixture_specs.py \
  --traces "$TRACES" --output specs/${SCENARIO}/closed_world.json \
  --config configs/mainline.yaml

for seed in 2025 2026 2027 2028 2029; do
  python scripts/train_window_predictor.py \
    --specs specs/${SCENARIO}/closed_world.json \
    --atoms cache/${SCENARIO}/trace_atoms.npz \
    --output-dir runs/${SCENARIO}/seed${seed} \
    --config configs/mainline.yaml --seed ${seed} --device cuda:0
done

python scripts/summarize_results.py \
  --runs-root runs/${SCENARIO} --output runs/${SCENARIO}/summary_overall.json
```

`metrics.json` reports Micro-F1 for each window size `m ∈ {1, ..., 5}` under
`test`; `summary_overall.json` aggregates the overall value over seeds.

### Reported closed-world Micro-F1 (%)

| Scenario | m=1 | m=2 | m=3 | m=4 | m=5 |
| --- | --- | --- | --- | --- | --- |
| Direct HTTPS | 99.35 | 98.76 | 96.71 | 97.70 | 96.78 |
| Trojan | 95.57 | 96.26 | 95.00 | 92.37 | 93.08 |
| VMess | 94.53 | 93.83 | 94.58 | 93.66 | 92.86 |

## 5. Open-world experiments (Fig. 3)

Background traffic is used for testing only.

```bash
# 5.1 Atom responses for unmonitored background traces.
python scripts/build_trace_atoms.py \
  --traces data/processed/open_world/background/traces.parquet \
  --encoder checkpoints/encoder/encoder.pth.tar \
  --vocabulary checkpoints/atom_vocabulary/${SCENARIO} \
  --output cache/${SCENARIO}/background_trace_atoms.npz \
  --config configs/mainline.yaml --device cuda:0

# 5.2 Target-present windows: m in 1..5 and b in 1..5, 500 windows per cell.
python scripts/build_open_world_specs.py \
  --monitored-traces data/processed/closed_world/${SCENARIO}/traces.parquet \
  --background-traces data/processed/open_world/background/traces.parquet \
  --source-specs specs/${SCENARIO}/closed_world.json \
  --output specs/${SCENARIO}/open_world.json --config configs/mainline.yaml

# 5.3 Frozen evaluation with the source-validation threshold.
for seed in 2025 2026 2027 2028 2029; do
  python scripts/evaluate_frozen.py \
    --source-run runs/${SCENARIO}/seed${seed} \
    --evaluation-specs specs/${SCENARIO}/open_world.json \
    --atoms cache/${SCENARIO}/trace_atoms.npz \
    --atoms cache/${SCENARIO}/background_trace_atoms.npz \
    --output runs/${SCENARIO}/seed${seed}/open_world.json --mode open_world
done
```

Each output file contains `overall` and per-cell groups named `m/b`
(e.g. `2/5`). Pooling `m ∈ {2, ..., 5}` at `b = 5` gives the values below.

### Reported open-world Micro-F1 (%) at b = 5, pooled m = 2–5

| Direct HTTPS | Trojan | VMess |
| --- | --- | --- |
| 90.58 | 91.70 | 88.42 |

The abstract reports the corresponding headline open-world scores of 92.37,
92.64 and 89.85 across the tested open-world settings.

## 6. Ablations

### Atom-count parameter `K` (Fig. 4)

Rebuild the vocabulary with a different number of requested clusters, then
re-extract and retrain. Only the vocabulary and its downstream caches change;
the encoder is frozen.

```bash
for K in 200 400 800 1200 1600 2400; do
  python scripts/build_atom_vocabulary.py \
    --traces data/processed/closed_world/direct/traces.parquet \
    --encoder checkpoints/encoder/encoder.pth.tar \
    --output-dir checkpoints/atom_vocabulary/direct_K${K} \
    --config configs/mainline.yaml --clusters ${K}
  python scripts/build_trace_atoms.py \
    --traces data/processed/closed_world/direct/traces.parquet \
    --encoder checkpoints/encoder/encoder.pth.tar \
    --vocabulary checkpoints/atom_vocabulary/direct_K${K} \
    --output cache/direct_K${K}/trace_atoms.npz --config configs/mainline.yaml
  python scripts/train_window_predictor.py \
    --specs specs/direct/closed_world.json \
    --atoms cache/direct_K${K}/trace_atoms.npz \
    --output-dir runs/direct_K${K}/seed2025 \
    --config configs/mainline.yaml --seed 2025
done
```

### Flow budget `k` per visit trace (Fig. 5)

Limit the flows retained per trace during Atom extraction. Sampling is
deterministic per trace (`seed + trace_id`), so the same budget always selects
the same flows.

```bash
for k in 1 2 4 8 16; do
  python scripts/build_trace_atoms.py \
    --traces data/processed/closed_world/direct/traces.parquet \
    --encoder checkpoints/encoder/encoder.pth.tar \
    --vocabulary checkpoints/atom_vocabulary/direct \
    --output cache/direct_k${k}/trace_atoms.npz \
    --config configs/mainline.yaml --max-flows-per-trace ${k}
  python scripts/train_window_predictor.py \
    --specs specs/direct/closed_world.json \
    --atoms cache/direct_k${k}/trace_atoms.npz \
    --output-dir runs/direct_k${k}/seed2025 \
    --config configs/mainline.yaml --seed 2025
done
```

The all-flow setting corresponds to no `--max-flows-per-trace` argument.

### Encoder source

The mainline uses an encoder pretrained on external unlabeled traffic. To
measure the contribution of that pretraining, replace the encoder checkpoint
with a randomly initialized one and repeat steps 4.1–4.3:

```bash
PYTHONPATH=src python - <<'PY'
import torch
from flowatom.models import DFMiniEncoder

torch.manual_seed(2025)
model = DFMiniEncoder(input_length=300)
torch.save(model.state_dict(), "checkpoints/encoder_random.pt")
PY
```

`load_pretrained_encoder` also accepts a plain feature state dictionary, so
`checkpoints/encoder_random.pt` can be passed through `--encoder` directly.

### Shared Atoms from pretraining traffic

To build Atoms from external unlabeled traffic instead of the target-protocol
training split:

```bash
python scripts/build_atom_vocabulary.py \
  --pretraining-parquet data/raw/pretraining/jp_mawi/signed_length_L300.parquet \
  --encoder checkpoints/encoder/encoder.pth.tar \
  --output-dir checkpoints/atom_vocabulary/shared \
  --config configs/mainline.yaml
```

## 7. Reproducibility checklist

- Fix `--seed` for every run; the paper uses `2025, 2026, 2027, 2028, 2029`.
- `metrics.json` records the resolved configuration, the best epoch, the
  validation-selected threshold and `test_used_for_selection: false`.
- Open-world and drift evaluations record `target_used_for_training: false`
  and `target_used_for_selection: false`.
- Standardization parameters come from training windows only.
- Sampling of flows for the flow-budget ablation is deterministic per trace.
- The synthetic smoke test uses the same code paths as the real pipeline, so a
  passing smoke test plus a green test suite validates the plumbing before
  launching a full campaign.
