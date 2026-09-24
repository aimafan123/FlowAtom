#!/usr/bin/env bash
# Exercise the public CLI on synthetic data without overwriting previous runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-configs/smoke.yaml}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
mkdir -p artifacts/smoke
WORK="${WORK:-$(mktemp -d "$ROOT/artifacts/smoke/run-XXXXXX")}"
if [[ -e "$WORK" ]] && { [[ ! -d "$WORK" ]] || [[ -n "$(ls -A "$WORK")" ]]; }; then
  echo "Smoke output must be new or empty: $WORK" >&2
  exit 2
fi
mkdir -p "$WORK"
"$PYTHON" scripts/make_synthetic_data.py --output-dir "$WORK/data" --sites 20 --pretraining-flows 3000
"$PYTHON" scripts/materialize_pretraining_shards.py \
  --input "$WORK/data/pretraining.parquet" --output-dir "$WORK/shards"
"$PYTHON" scripts/pretrain_encoder.py \
  --manifest "$WORK/shards/manifest.json" --output-dir "$WORK/encoder" \
  --config "$CONFIG" --workers 0 --device cpu
"$PYTHON" scripts/run_experiment.py \
  --traces "$WORK/data/traces.parquet" \
  --background-traces "$WORK/data/background_traces.parquet" \
  --encoder "$WORK/encoder/encoder.pth.tar" --output-dir "$WORK/experiment" \
  --config "$CONFIG" --seed 2025 --device cpu
printf '\nSMOKE TEST PASSED. Artifacts under %s\n' "$WORK"
