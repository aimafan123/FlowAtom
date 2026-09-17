# Scripts

Every script is a thin, deterministic command-line wrapper around
`src/flowatom`. Run them from the repository root; each script bootstraps
`src/` onto `sys.path`, so `PYTHONPATH` is optional.

| Script | Purpose |
| --- | --- |
| `make_synthetic_data.py` | Generate a small synthetic dataset in the FlowAtom schema. |
| `materialize_pretraining_shards.py` | Convert a fixed-sequence parquet into mmap-friendly NPY shards. |
| `pretrain_encoder.py` | Train the MoCo flow encoder. |
| `build_atom_vocabulary.py` | Build a scenario-specific Atom vocabulary (k-means + XGBoost mapper). |
| `build_trace_atoms.py` | Extract per-trace Atom responses. |
| `build_mixture_specs.py` | Build closed-world window specifications. |
| `build_open_world_specs.py` | Build target-present open-world window specifications. |
| `train_window_predictor.py` | Train one website-set predictor seed. |
| `evaluate_frozen.py` | Evaluate a frozen predictor on open-world windows. |
| `summarize_results.py` | Aggregate Micro-F1 across seeds. |
| `run_smoke_test.sh` | Run the whole synthetic pipeline end to end. |

## Typical order

```
make_synthetic_data.py            (or convert real captures)
materialize_pretraining_shards.py
pretrain_encoder.py
build_atom_vocabulary.py
build_trace_atoms.py              (monitored traces)
build_mixture_specs.py
train_window_predictor.py         (once per seed)
build_trace_atoms.py              (background traces)
build_open_world_specs.py
evaluate_frozen.py
summarize_results.py
```

Every script prints a JSON summary to stdout. Errors are raised rather than
silently skipped: for example, a flow below the minimum nonzero payload packet
count aborts extraction instead of being dropped silently.

See [`../docs/reproduction.md`](../docs/reproduction.md) for the full command
sequence and paper-level parameters.
