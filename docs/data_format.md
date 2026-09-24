# Data formats

This page documents the file formats passed between pipeline stages.

## 1. Trace parquet

One row per visit trace. Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `trace_id` | int64 | Globally unique trace identifier within a scenario. |
| `label` | int64 | Monitored website label, or `-1` for background traces. |
| `site` | string | Website name; each monitored label maps to exactly one site. |
| `split` | string | `train` or `test`. Validation windows are carved out of `train` traces. |
| `test_order` | int64 | Deterministic ordering of test traces within a label. |
| `payload_flows` | list<list<int32>> | Per flow, the payload length of each packet in order. |
| `direction_flows` | list<list<int8>> | Per flow, `+1` for downlink and `-1` for uplink packets. |

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

Short flows are allowed in this raw trace format. Both Atom construction and
response extraction drop zero-payload packets, then skip flows below the
configured minimum packet count. Flow budgets apply after this filtering.
A trace whose flows are all filtered retains its ID and label and has zero
responses. Packet/flow misalignment remains an error. Multiple background
sites may share label `-1`; the one-label/one-site constraint applies only to
monitored labels.

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

Closed-world specs also contain `trace_pools`, with complete sorted trace-ID
lists under `train`, `validation` and `test`, including traces not sampled into
any window. The CLI writes `trace_dataset_sha256` for the input parquet.
Generate this file before fitting Atoms; the Atom builder checks its source
hash, pool membership and disjointness. Training rechecks the pools and Atom
fit provenance rather than trusting protocol flags.

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
| `flow_count` | `[traces]` | Integer count of flows actually contributing after short-flow filtering, budget sampling and confidence filtering. |
| `metadata` | `[1]` | Schema-v2 JSON with `contract`, `atom_fit`, `source_sha256`, and raw/short/retained flow totals. |

`contract` contains `encoder_sha256`, `vocabulary_sha256`, `representation`,
`input_length`, `minimum_nonzero_payload_packets`, `confidence_threshold`,
`max_flows_per_trace` and `flow_sampling_seed` (null without a budget).
Tables may be combined only when the entire contract and Atom fit provenance
match; equal dimensions alone are insufficient. Duplicate trace IDs are rejected.
`total_flow_count` counts raw flows, `short_flow_count` counts those below the
packet threshold, and `retained_flow_count` equals the sum of `flow_count`.
Schema-v1 caches must be regenerated.

Because `peak` is already the per-trace maximum over flows, max pooling over
the traces of a window equals max pooling over every flow in the window.

## 5. Atom vocabulary directory

Produced by `scripts/build_atom_vocabulary.py`:

| File | Meaning |
| --- | --- |
| `scaler.joblib` | `StandardScaler` on embeddings; absent when scaling is disabled. |
| `atom_model.json` | XGBoost mapper. |
| `clusters.npz` | k-means centers, cluster counts and retained cluster indices. |
| `metadata.json` | Schema v2, dimensions, model/scaler hashes, vocabulary identity and fit provenance. |
| `build_metadata.json` | Full construction record: seed, flow count, cluster statistics, source. |

The vocabulary identity hashes its model/scaler file hashes and provenance.
Provenance records the encoder hash, preprocessing contract, source-data hash
and Atom fit source. For scenario-specific Atoms it also records the exact
training pool and the fingerprint of all three trace pools.

With `--keep-embeddings`, `embeddings.npy` is accompanied by
`embeddings.npy.json`. Reuse checks the array shape, dtype, content hash and
source/configuration contract. Incomplete caches cannot be silently reused;
`--rebuild-embeddings` regenerates the array and sidecar.

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
and classification heads are ignored. If the checkpoint records `input_length`,
a conflicting requested length is rejected. For legacy plain state dictionaries
without length metadata, the caller-supplied length remains the explicit contract.
The file content hash identifies the encoder throughout downstream artifacts.

## 7. Trained run directory

`scripts/train_window_predictor.py` writes:

| File | Meaning |
| --- | --- |
| `metrics.json` | Labels, input/Atom dimensions, validation metrics, per-size test metrics, config, protocol flags and the full epoch history. |
| `model.pt` | PyTorch state dictionary of the window-set MLP. |
| `standardizer.npz` | `mean` and `scale` arrays fitted on training windows. |
| `artifacts.json` | Content hashes binding `model.pt`, `standardizer.npz` and `metrics.json` to this run. |

`metrics.json` also stores the cache feature contract, Atom fit provenance and
trace-pool fingerprint. Frozen evaluation requires an identical feature
contract; loading rejects missing manifests or mixed/tampered run files.
Legacy runs require retraining after rebuilding their upstream artifacts.

`scripts/evaluate_frozen.py` verifies the manifest, reads the run files, freezes the
validation-selected threshold, and writes a metrics JSON with `overall` and
`groups` sections.

## 8. Synthetic data

`scripts/make_synthetic_data.py` writes the same trace and pretraining parquet
schemas to `data/generated/synthetic/`:

- `traces.parquet`: monitored train/test traces.
- `background_traces.parquet`: unmonitored traces with `label = -1`.
- `pretraining.parquet`: fixed-length signed sequences.
- `synthetic_summary.json`: generation parameters.
