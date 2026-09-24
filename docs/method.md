# Method

FlowAtom identifies the set of monitored websites present in an **observation
window**: an unordered set of complete flows observed on the client egress
link. The model never receives visit boundaries, flow-to-visit assignments,
flow-to-website assignments, or the true number of websites.

The pipeline has three stages. Stage 1 and stage 2 use no website labels;
window labels supervise only stage 3.

## 1. Flow representation pretraining

Each flow is a packet sequence with the payload length and the packet
direction. FlowAtom drops zero-payload packets and builds a signed
payload-length sequence

```
x_i = [ abs(l_1) * s_1, abs(l_2) * s_2, ..., abs(l_n) * s_n ]
```

where `s_j ∈ {+1, -1}` is the packet direction. Sequences are truncated or
right-padded with zeros to `L = 300`, and flows with fewer than 10 nonzero
payload packets are discarded. Filtering happens before any per-trace flow
budget is applied. Traces with no eligible flows keep their ID and label and
receive an all-zero Atom response vector; they are not removed from evaluation.
Malformed packet/flow alignment still raises an error.

A DF-mini convolutional encoder `E` maps the sequence to a flat embedding
`z_i = E(x_i)`. The encoder has three convolutional blocks (16, 32, 64
channels), each with two `kernel_size=8, padding=same` convolutions followed by
batch normalization, a non-linearity, a `MaxPool1d(4, 2, 2)` and dropout. For
`L = 300` the flattened embedding has 2496 dimensions.

The encoder is pretrained on large-scale external unlabeled traffic with
Momentum Contrast (MoCo):

- a queue of 65536 normalized negative embeddings,
- momentum `m = 0.999` for the key encoder,
- temperature `τ = 0.07`,
- SGD with learning rate `0.03`, momentum `0.9`, weight decay `1e-4`, batch
  size 8192, and a 0.1× learning-rate decay at epochs 60 and 80.

Contrastive views are produced by TCP-aware augmentations: `copyrto`,
`copyfr`, `shiftrto`, `shiftfr` and packet-size variation. The default policy
`copyrto` uses the original sequence as the first view and an augmented copy as
the second view. After pretraining the projection head is discarded and the
momentum encoder `E` is frozen.

Implementation: `flowatom.representation`, `flowatom.models.DFMiniEncoder`,
`flowatom.models.load_pretrained_encoder`, `flowatom.pretraining`.

## 2. Atom construction

First generate closed-world specs, including explicit, disjoint `trace_pools`
for training, validation and testing. For each website, at least two original
training traces are required: 90% (rounded down, at least one) enter the training
pool and the rest enter validation. Atom construction requires these specs and
embeds only eligible flows from the **training pool**. Validation and test traces
are excluded from clustering and mapper fitting. Website labels are used to
stratify the split, but are never Atom training targets:

1. Optionally standardize the embeddings. The paper mainline uses raw
   embeddings (`embedding_scaling: false`).
2. Run k-means with `K = 1200` requested clusters, `n_init = 3` and
   `batch_size = 8192`.
3. Discard clusters with fewer than 50 members. The centers of the retained
   clusters are the **Atoms** `A = {a_1, ..., a_A}`; typically `A ≈ 1189`.
4. Use the cluster assignments as pseudo-labels to train an XGBoost mapper `M`
   with 100 trees, maximum depth 4, learning rate 0.1, subsample 0.8 and
   column subsample 0.8. At most 100000 flows are sampled for the mapper, and
   at least one representative of every retained cluster is always included.

The mapper produces a soft Atom response vector `q_i = M(z_i)` where `q_{i,a}`
is the score of Atom `a` for flow `i`. k-means is used only for offline Atom
discovery; at inference time the frozen encoder and mapper directly produce
Atom responses.

Implementation: `flowatom.atoms.build_atom_vocabulary`,
`flowatom.atoms.AtomVocabulary`.

## 3. Window representation and website-set prediction

A window `T` is a set of complete visit traces; its label is the set of
monitored websites those traces visit. For every flow of every trace in the
window, the frozen encoder and mapper compute Atom responses. A per-Atom max
pooling aggregates them:

```
Φ_a(T) = max_{f ∈ T} q_{f,a}
```

The result `Φ(T) ∈ R^A` is fixed-dimensional and invariant to flow order.
Padding a window with more flows can only increase an Atom response, which is
exactly the intended "any flow provides this evidence" semantics.

Standardization parameters (per-Atom mean and standard deviation) are fitted on
downstream **training** windows only and applied to validation, test and target
windows. A two-hidden-layer MLP (512 and 256 units, batch normalization, GELU,
dropout 0.2) outputs one logit per monitored website and is trained with
positive-weighted binary cross-entropy:

```
w_c = (N - n_c) / max(n_c, 1)
```

with AdamW (learning rate `1e-3`, weight decay `1e-4`), batch size 512, up to
60 epochs and early stopping with patience 10 after a minimum of 10 epochs.

At inference, a website is predicted when its sigmoid score reaches a threshold
chosen on the validation split over the grid
`{0.05, 0.075, ..., 0.95}`; at most five websites are decoded per window. Test
or target traffic is never used for model or threshold selection.

Implementation: `flowatom.window`, `flowatom.training.train_window_predictor`,
`flowatom.evaluation`.

## Threat model

FlowAtom targets Direct HTTPS and non-multiplexed encrypted proxies, where the
attacker observes the client egress link (Direct HTTPS) or the client–proxy link
(Trojan, VMess) and can separate connections with five-tuples or equivalent
identifiers. Only packet payload lengths and directions are used. Connection
identifiers are used for flow separation; IP addresses, DNS records and SNI are
never part of the model input.

## Hyperparameter summary

| Parameter | Value |
| --- | --- |
| Input representation | signed payload length |
| Sequence length `L` | 300 |
| Minimum nonzero payload packets | 10 |
| Encoder embedding dimension | 2496 |
| MoCo feature dimension / queue | 128 / 65536 |
| MoCo momentum / temperature | 0.999 / 0.07 |
| Pretraining epochs / batch size | 45 / 8192 |
| Pretraining learning rate | 0.03 (×0.1 at epochs 60, 80) |
| Requested Atoms `K` | 1200 |
| Minimum cluster size | 50 |
| k-means `n_init` / batch size | 3 / 8192 |
| XGBoost trees / depth / learning rate | 100 / 4 / 0.1 |
| Mapper sample size | 100000 |
| Flow confidence threshold | 0.0 |
| Window aggregation | per-Atom max pooling |
| Decoder hidden layers | 512, 256 |
| Decoder dropout | 0.2 |
| Decoder learning rate / weight decay | 1e-3 / 1e-4 |
| Decoder epochs / patience | 60 / 10 |
| Max decoded websites | 5 |
| Reported metric | Micro-F1 over five seeds |

## Evaluation protocols

**Closed world.** Single-website visit traces are split into disjoint training,
validation and test sets. Windows are built offline by mixing complete traces
within a split, with `m ∈ {1, 2, 3, 4, 5}` distinct monitored websites.
The default 15,000 training and 1,500 validation windows are totals across
the five sizes (3,000 and 300 per size respectively). The same
source-trained model is evaluated on every window size.

**Open world.** Background (unmonitored) traffic is added to windows that
contain monitored visits. Evaluation is grouped by the number of monitored
websites `m ∈ {1, ..., 5}` and the number of background traces
`b ∈ {1, ..., 5}`, with 500 windows per cell. Background traffic is used for
testing only.

**Ablations.** The Atom count `K` is varied by rebuilding the vocabulary with a
different `--clusters` value. The per-visit flow budget `k` is varied with
`--max-flows-per-trace k` during trace Atom extraction, then retraining the
window predictor on the resulting cache. Both are described in
[`reproduction.md`](reproduction.md).

## Metric

Micro-F1 pools true positives, false positives and false negatives over all
windows before computing precision, recall and their harmonic mean. It is
reported as `f1` / `micro_f1` in every metrics file.
