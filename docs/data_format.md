# Data formats

This page documents every file format that crosses a pipeline boundary. All
formats are deliberately simple so that captures can be converted from other
toolchains with little effort.

## 1. Trace parquet

One row per visit trace. Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `trace_id` | int64 | Globally unique trace identifier within a scenario. |
| `label` | int64 | Monitored website label, or `-1` for background traces. |
| `site` | string | Website name; a label maps to exactly one site. |
| `split` | string | `train` or `test`. Validation windows are carved out of `train` traces. |
| `test_order` | int64 | Deterministic ordering of test traces within a label. |
| `payload_flows` | list<list<int32>> | Per flow, the payload length of each packet in order. |
| `direction_flows` | list<list<int8>> | Per flow, `+1` for downlink and `-1` for uplink packets. |

Optional columns such as `timestamp_flows` (list<list<float>>) are preserved
but not required by the model. Packet collections of a trace must all have the
same number of flows, and the packet collections of a flow must align.

Example (JSON view of one row):

```json
{
  "trace_id": 42,
  "label": 7,
  "site": "example.com",
  "split": "train",
  "test_order": 0,
  "payload_flows": [[0, 517, 0, 1400, 220], [88, 0, 1300]],
  "direction_flows": [[1, 1, -1, 1, -1], [-1, 1, 1]]
}
```

Identity-related fields (IP addresses, DNS, SNI, ports) must not be included:
FlowAtom consumes only payload lengths and directions.

## 2. Pretraining parquet

Fixed-length signed sequences used for MoCo pretraining. The same file may
carry three equivalent columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `signed_length` | list<int32> | `abs(payload) * sign(direction)`, padding `0`. |
| `payload_length` | list<int32> | Absolute payload lengths with padding `0`. |
| `direction` | list<int32> | `+1`/`-1` directions with padding `0`. |

All rows must have the same list length, equal to the configured
`input_length` (300 in the paper). Sequences are already filtered to satisfy
the minimum nonzero payload packet count.

## 3. Window specification JSON

Built by `scripts/build_mixture_specs.py` (closed world) and
`scripts/build_open_world_specs.py` (open world). Top-level fields:

```json
{
  "seed": 2025,
  "labels": [0, 1, 2],
  "train": [ ... ],
  "validation": [ ... ],
  "test": [ ... ],
  "protocol": { ... }
}
```

A closed-world window element:

```json
{"websites_num": 2, "labels": [0, 2], "trace_ids": [11, 57]}
```

An open-world window element additionally records the monitored and background
counts; background trace ids are appended to `trace_ids`:

```json
{
  "websites_num": 2,
  "monitored_websites_num": 2,
  "background_traces_num": 3,
  "labels": [0, 2],
  "trace_ids": [11, 57, 9001, 9002, 9003]
}
```

`websites_num` equals the number of monitored labels. The true website count is
never passed to the model; it is used only to group metrics.

The `protocol` block records invariants such as `trace_disjoint`,
`test_used_for_selection: false`, `background_used_for_training: false` and
`background_used_for_selection: false`.

## 4. Trace Atom cache (`.npz`)

Produced by `scripts/build_trace_atoms.py` and loaded by
`flowatom.atoms.TraceAtomTable`. Arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `trace_ids` | `[traces]` | Trace identifiers. |
| `labels` | `[traces]` | Website labels (`-1` for background). |
| `peak` | `[traces, atoms]` | Maximum Atom response over the retained flows of each trace. |
| `flow_count` | `[traces]` | Number of retained flows per trace. |
| `confidence_threshold` | `[1]` | Flow retention threshold. |
| `metadata` | `[1]` | JSON with `representation` and `input_length`. |

Because `peak` is already the per-trace maximum over flows, max pooling over
the traces of a window equals max pooling over every flow in the window.

## 5. Atom vocabulary directory

Produced by `scripts/build_atom_vocabulary.py`:

| File | Meaning |
| --- | --- |
| `scaler.joblib` | `StandardScaler` on embeddings; absent when scaling is disabled. |
| `atom_model.json` | XGBoost mapper. |
| `clusters.npz` | k-means centers, cluster counts and retained cluster indices. |
| `metadata.json` | Schema version, classifier type, embedding/Atom dimensions. |
| `build_metadata.json` | Full construction record: seed, flow count, cluster statistics, source. |

## 6. Encoder checkpoint

`scripts/pretrain_encoder.py` writes `encoder.pth.tar`:

```python
{
  "arch": "DFMiniEncoder",
  "input_length": 300,
  "feature_dim": 128,
  "queue_size": 65536,
  "momentum": 0.999,
  "temperature": 0.07,
  "state_dict": { ... "encoder_k.features...": tensor, ... },
  "epoch": 45,
  "history": [ ... ]
}
```

`flowatom.models.load_pretrained_encoder` also accepts a plain feature-layer
state dictionary and legacy key layouts (`feature_layers.*`, `block1.*`,
`block2.*`, `block3.*`). Only convolutional feature layers are read; projection
and classification heads are ignored.

## 7. Trained run directory

`scripts/train_window_predictor.py` writes:

| File | Meaning |
| --- | --- |
| `metrics.json` | Labels, input/Atom dimensions, validation metrics, per-size test metrics, config, protocol flags and the full epoch history. |
| `model.pt` | PyTorch state dictionary of the window-set MLP. |
| `standardizer.npz` | `mean` and `scale` arrays fitted on training windows. |

`scripts/evaluate_frozen.py` reads these three files, freezes the
validation-selected threshold, and writes a metrics JSON with `overall` and
`groups` sections.

## 8. Synthetic data

`scripts/make_synthetic_data.py` writes the same trace and pretraining parquet
schemas to `data/generated/synthetic/`:

- `traces.parquet`: monitored train/test traces.
- `background_traces.parquet`: unmonitored traces with `label = -1`.
- `pretraining.parquet`: fixed-length signed sequences.
- `synthetic_summary.json`: generation parameters.
