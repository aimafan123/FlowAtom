# Command-line reference

Install the package and use `flowatom <command> --help`, or run
`python scripts/<name>.py --help` from a source checkout. Both routes call the
same implementation in `src/flowatom/cli/`. `python -m flowatom` is equivalent
to the installed console command.

| Installed command | Source wrapper | Purpose |
| --- | --- | --- |
| `flowatom run` | `run_experiment.py` | One complete downstream scenario/seed |
| `flowatom make-synthetic-data` | `make_synthetic_data.py` | Synthetic trace and pretraining data |
| `flowatom materialize-pretraining-shards` | `materialize_pretraining_shards.py` | Mmap-friendly flow shards |
| `flowatom pretrain-encoder` | `pretrain_encoder.py` | MoCo flow encoder |
| `flowatom build-mixture-specs` | `build_mixture_specs.py` | Disjoint trace pools and closed-world windows |
| `flowatom build-atom-vocabulary` | `build_atom_vocabulary.py` | Training-pool Atoms and mapper |
| `flowatom build-trace-atoms` | `build_trace_atoms.py` | Trace-level Atom responses |
| `flowatom train-window-predictor` | `train_window_predictor.py` | Standardizer, predictor and validation-selected decoder |
| `flowatom build-open-world-specs` | `build_open_world_specs.py` | Target-present test windows |
| `flowatom evaluate-frozen` | `evaluate_frozen.py` | Frozen closed/open-world evaluation |
| `flowatom summarize-results` | `summarize_results.py` | Aggregate per-run Micro-F1 |

`run_smoke_test.sh` generates synthetic data, pretrains a small encoder and calls
`flowatom run` through its source wrapper. It creates a new output directory
by default. Set `WORK` to a new or empty directory to choose the location;
set `PYTHON` to the intended interpreter.

## Dependencies between stages

```text
external pretraining flows -> shards -> encoder
monitored traces -> disjoint closed-world specs
training pool + encoder -> Atom vocabulary
traces + encoder + vocabulary -> trace responses
training/validation windows + trace responses -> frozen predictor
held-out/background windows + matching responses -> evaluation
```

`build-atom-vocabulary --traces` requires `--specs`; it reads only the declared
training pool. Short flows are filtered before embedding and flow-budget
sampling. Zero-evidence traces remain represented. Artifact contracts detect
incompatible encoders, vocabularies, preprocessing, caches and predictors.

The full runner records progress in its manifest and writes one stage log per
command. Individual stages may print progress before a final JSON summary.
Use output files for machine-readable results rather than treating all stdout
as a JSON document. See [reproduction](../docs/reproduction.md) for examples.
