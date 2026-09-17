#!/usr/bin/env bash
# End-to-end FlowAtom smoke test on synthetic data.
#
# Usage:
#   PYTHON=.venv/bin/python bash scripts/run_smoke_test.sh
#
# It runs the complete pipeline: synthetic data -> pretraining shards ->
# MoCo encoder -> Atom vocabulary -> trace Atom responses -> closed-world
# window training -> open-world frozen evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-configs/smoke.yaml}"

DATA="data/generated/synthetic"
WORK="artifacts/smoke"
rm -rf "$WORK"

echo "== 1/8 generate synthetic data =="
"$PYTHON" scripts/make_synthetic_data.py --output-dir "$DATA" --sites 20 --pretraining-flows 3000

echo "== 2/8 materialize pretraining shards =="
"$PYTHON" scripts/materialize_pretraining_shards.py \
  --input "$DATA/pretraining.parquet" --output-dir "$WORK/shards"

echo "== 3/8 pretrain MoCo encoder =="
"$PYTHON" scripts/pretrain_encoder.py \
  --manifest "$WORK/shards/manifest.json" --output-dir "$WORK/encoder" \
  --config "$CONFIG" --workers 0

echo "== 4/8 build Atom vocabulary =="
"$PYTHON" scripts/build_atom_vocabulary.py \
  --traces "$DATA/traces.parquet" \
  --encoder "$WORK/encoder/encoder.pth.tar" \
  --output-dir "$WORK/atom_vocabulary" --config "$CONFIG"

echo "== 5/8 extract trace Atom responses =="
"$PYTHON" scripts/build_trace_atoms.py \
  --traces "$DATA/traces.parquet" \
  --encoder "$WORK/encoder/encoder.pth.tar" \
  --vocabulary "$WORK/atom_vocabulary" \
  --output "$WORK/trace_atoms.npz" --config "$CONFIG"

"$PYTHON" scripts/build_trace_atoms.py \
  --traces "$DATA/background_traces.parquet" \
  --encoder "$WORK/encoder/encoder.pth.tar" \
  --vocabulary "$WORK/atom_vocabulary" \
  --output "$WORK/background_trace_atoms.npz" --config "$CONFIG"

echo "== 6/8 build closed-world window specs =="
"$PYTHON" scripts/build_mixture_specs.py \
  --traces "$DATA/traces.parquet" --output "$WORK/specs/closed_world.json" --config "$CONFIG"

echo "== 7/8 train window-set predictor =="
"$PYTHON" scripts/train_window_predictor.py \
  --specs "$WORK/specs/closed_world.json" --atoms "$WORK/trace_atoms.npz" \
  --output-dir "$WORK/run/seed2025" --config "$CONFIG" --seed 2025

echo "== 8/8 open-world frozen evaluation =="
"$PYTHON" scripts/build_open_world_specs.py \
  --monitored-traces "$DATA/traces.parquet" \
  --background-traces "$DATA/background_traces.parquet" \
  --source-specs "$WORK/specs/closed_world.json" \
  --output "$WORK/specs/open_world.json" --config "$CONFIG"
"$PYTHON" scripts/evaluate_frozen.py \
  --source-run "$WORK/run/seed2025" \
  --evaluation-specs "$WORK/specs/open_world.json" \
  --atoms "$WORK/trace_atoms.npz" --atoms "$WORK/background_trace_atoms.npz" \
  --output "$WORK/open_world_metrics.json" --mode open_world

echo
echo "SMOKE TEST PASSED. Artifacts under $WORK"
