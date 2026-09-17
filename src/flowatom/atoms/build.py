"""Construct an Atom vocabulary from frozen flow embeddings.

Atoms are the centers of retained k-means clusters of training-flow
representations. The cluster assignments are then used as pseudo-labels to
train a gradient-boosted mapper that turns a flow embedding into a soft Atom
response vector. No website labels are used at any point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from flowatom.atoms.vocabulary import AtomVocabulary, AtomVocabularyError

PathLike = Union[str, Path]

DEFAULT_XGBOOST_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def classifier_fit_indices(
    effective_labels: np.ndarray, sample_size: int, seed: int
) -> np.ndarray:
    """Sample mapper training flows while guaranteeing every Atom is present."""

    effective_labels = np.asarray(effective_labels, dtype=np.int64)
    eligible = np.flatnonzero(effective_labels >= 0)
    if not len(eligible):
        raise AtomVocabularyError("Atom vocabulary has no eligible mapper samples")
    rng = np.random.default_rng(seed)
    size = len(eligible) if sample_size <= 0 else min(sample_size, len(eligible))
    sampled = rng.choice(eligible, size=size, replace=False)
    classes, first = np.unique(effective_labels[eligible], return_index=True)
    expected = np.arange(int(effective_labels.max()) + 1)
    if not np.array_equal(classes, expected):
        raise AtomVocabularyError("effective Atom labels are not contiguous")
    representatives = eligible[first]
    return np.unique(np.concatenate((sampled, representatives)))


def build_atom_vocabulary(
    embeddings: np.ndarray,
    output_dir: PathLike,
    *,
    clusters: int = 1200,
    minimum_cluster_size: int = 50,
    scaling: bool = False,
    kmeans_n_init: int = 3,
    kmeans_batch_size: int = 8192,
    xgb_sample_size: int = 100000,
    xgboost_params: Optional[Mapping[str, Any]] = None,
    seed: int = 2025,
    tree_device: str = "auto",
    transform_chunk_size: int = 16384,
) -> Tuple[AtomVocabulary, Dict[str, Any]]:
    """Fit an Atom vocabulary and persist it to ``output_dir``.

    Returns the vocabulary and a metadata dictionary describing the fit.
    ``embeddings`` may be a memory-mapped ``.npy`` array; standardization and
    k-means then stream over it in chunks instead of materializing a second
    full-size array in RAM.
    """

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise AtomVocabularyError("embeddings must be a non-empty matrix")
    if clusters <= 0:
        raise AtomVocabularyError("clusters must be positive")
    if clusters > len(embeddings):
        raise AtomVocabularyError(
            f"requested {clusters} Atom clusters for only {len(embeddings)} flows"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_normalized: Optional[Path] = None
    if scaling:
        scaler = StandardScaler()
        for start in range(0, len(embeddings), transform_chunk_size):
            scaler.partial_fit(embeddings[start : start + transform_chunk_size])
        temporary_normalized = output_dir / "normalized_embeddings.npy"
        normalized = np.lib.format.open_memmap(
            temporary_normalized,
            mode="w+",
            dtype=np.float32,
            shape=embeddings.shape,
        )
        for start in range(0, len(embeddings), transform_chunk_size):
            stop = min(start + transform_chunk_size, len(embeddings))
            normalized[start:stop] = scaler.transform(embeddings[start:stop])
        normalized.flush()
    else:
        scaler = None
        normalized = embeddings

    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        n_init=kmeans_n_init,
        batch_size=kmeans_batch_size,
    )
    cluster_labels = kmeans.fit_predict(normalized)
    counts = np.bincount(cluster_labels, minlength=clusters)
    valid_clusters = np.flatnonzero(counts >= minimum_cluster_size)
    if not len(valid_clusters):
        raise AtomVocabularyError(
            f"no cluster reaches the minimum size of {minimum_cluster_size}"
        )
    remap = np.full(clusters, -1, dtype=np.int64)
    remap[valid_clusters] = np.arange(len(valid_clusters))
    effective_labels = remap[cluster_labels]

    fit_indices = classifier_fit_indices(effective_labels, xgb_sample_size, seed)
    params = dict(DEFAULT_XGBOOST_PARAMS)
    params.update(xgboost_params or {})
    if tree_device == "auto":
        try:
            import torch

            tree_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover - torch is a hard dependency
            tree_device = "cpu"
    classifier = XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
        tree_method="hist",
        device=tree_device,
        **params,
    )
    classifier.fit(normalized[fit_indices], effective_labels[fit_indices])

    vocabulary = AtomVocabulary(classifier=classifier, scaler=scaler, classifier_type="xgboost")
    probabilities = counts / counts.sum()
    used = probabilities > 0
    entropy = float(-(probabilities[used] * np.log(probabilities[used])).sum())
    np.savez_compressed(
        output_dir / "clusters.npz",
        cluster_centers=kmeans.cluster_centers_.astype(np.float32),
        cluster_counts=counts.astype(np.int64),
        valid_clusters=valid_clusters.astype(np.int64),
    )
    metadata: Dict[str, Any] = {
        "seed": seed,
        "training_flows": int(len(embeddings)),
        "embedding_dimension": int(embeddings.shape[1]),
        "scaling_enabled": bool(scaling),
        "requested_clusters": int(clusters),
        "effective_atoms": int(len(valid_clusters)),
        "minimum_cluster_size": int(minimum_cluster_size),
        "cluster_size_min": int(counts.min()),
        "cluster_size_max": int(counts.max()),
        "cluster_size_median": float(np.median(counts)),
        "kmeans_inertia": float(kmeans.inertia_),
        "normalized_usage_entropy": entropy / float(np.log(clusters)),
        "mapper_training_samples": int(len(fit_indices)),
        "xgboost": params,
        "tree_device": tree_device,
        "clusters_file": "clusters.npz",
    }
    vocabulary.save(output_dir, metadata=metadata)
    if temporary_normalized is not None:
        del normalized
        temporary_normalized.unlink(missing_ok=True)
    return vocabulary, metadata
