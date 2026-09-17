"""Embedding-to-Atom probability inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

PathLike = Union[str, Path]
SCALER_FILE = "scaler.joblib"
XGBOOST_FILE = "atom_model.json"
CENTROID_FILE = "atom_centroids.npz"
METADATA_FILE = "metadata.json"


class AtomVocabularyError(ValueError):
    """Raised when embeddings or Atom artifacts violate the contract."""


class NearestCentroidClassifier:
    """Expose exact nearest-valid-centroid assignments as class probabilities."""

    def __init__(self, centers: np.ndarray) -> None:
        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim != 2 or len(centers) == 0:
            raise AtomVocabularyError("centers must be a non-empty matrix")
        self.centers = centers
        self.classes_ = np.arange(len(centers), dtype=np.int64)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.metrics import pairwise_distances_argmin

        values = np.asarray(embeddings, dtype=np.float32)
        assignments = pairwise_distances_argmin(values, self.centers, metric="euclidean")
        probabilities = np.zeros((len(values), len(self.centers)), dtype=np.float32)
        probabilities[np.arange(len(values)), assignments] = 1.0
        return probabilities


@dataclass
class AtomVocabulary:
    """Pair an optional embedding scaler with an Atom classifier."""

    classifier: Any
    scaler: Optional[Any] = None
    classifier_type: str = "xgboost"
    embedding_dimension: Optional[int] = None

    def __post_init__(self) -> None:
        if self.embedding_dimension is None:
            if self.scaler is not None:
                self.embedding_dimension = int(len(self.scaler.mean_))
            else:
                self.embedding_dimension = int(
                    getattr(self.classifier, "n_features_in_", 0)
                )
        self.embedding_dimension = int(self.embedding_dimension)
        if self.scaler is None and self.classifier_type == "nearest_centroid":
            self.embedding_dimension = int(self.classifier.centers.shape[1])
        self.atom_count = int(len(self.classifier.classes_))
        if self.embedding_dimension <= 0:
            raise AtomVocabularyError("cannot infer the embedding dimension")
        if self.atom_count <= 0:
            raise AtomVocabularyError("the Atom classifier has no classes")

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2:
            raise AtomVocabularyError("embeddings must have shape [flows, features]")
        if values.shape[1] != self.embedding_dimension:
            raise AtomVocabularyError(
                f"embedding dimension {values.shape[1]} does not match "
                f"vocabulary dimension {self.embedding_dimension}"
            )
        if self.scaler is None:
            return values
        return np.asarray(self.scaler.transform(values), dtype=np.float32)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(
            self.classifier.predict_proba(self.transform(embeddings)), dtype=np.float32
        )
        expected = (len(np.asarray(embeddings)), self.atom_count)
        if probabilities.shape != expected:
            raise AtomVocabularyError(
                f"Atom probability shape {probabilities.shape} != {expected}"
            )
        return probabilities

    def save(self, output_dir: PathLike, metadata: Optional[dict] = None) -> Path:
        """Write the scaler, Atom classifier and metadata to a directory."""

        import joblib

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.scaler is not None:
            joblib.dump(self.scaler, output_dir / SCALER_FILE)
        if self.classifier_type == "xgboost":
            self.classifier.save_model(output_dir / XGBOOST_FILE)
        else:
            centers = np.asarray(self.classifier.centers, dtype=np.float32)
            np.savez_compressed(output_dir / CENTROID_FILE, cluster_centers=centers)
        payload = {
            "schema_version": 1,
            "classifier_type": self.classifier_type,
            "scaling_enabled": self.scaler is not None,
            "embedding_dimension": self.embedding_dimension,
            "atom_count": self.atom_count,
        }
        payload.update(metadata or {})
        (output_dir / METADATA_FILE).write_text(json.dumps(payload, indent=2))
        return output_dir

    @classmethod
    def load(cls, output_dir: PathLike) -> "AtomVocabulary":
        import joblib
        from xgboost import XGBClassifier

        output_dir = Path(output_dir)
        metadata_path = output_dir / METADATA_FILE
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text())
        classifier_type = str(metadata.get("classifier_type", "xgboost"))
        scaler = None
        if metadata.get("scaling_enabled", True):
            scaler_path = output_dir / SCALER_FILE
            if not scaler_path.is_file():
                raise FileNotFoundError(scaler_path)
            scaler = joblib.load(scaler_path)
        if classifier_type == "xgboost":
            model_path = output_dir / XGBOOST_FILE
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            classifier = XGBClassifier(tree_method="hist", device="cpu")
            classifier.load_model(model_path)
        elif classifier_type == "nearest_centroid":
            centroid_path = output_dir / CENTROID_FILE
            if not centroid_path.is_file():
                raise FileNotFoundError(centroid_path)
            with np.load(centroid_path) as saved:
                classifier = NearestCentroidClassifier(saved["cluster_centers"])
        else:
            raise AtomVocabularyError(f"unsupported classifier type: {classifier_type}")
        return cls(classifier=classifier, scaler=scaler, classifier_type=classifier_type)
