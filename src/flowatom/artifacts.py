"""Content fingerprints and explicit contracts for pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


class ArtifactError(ValueError):
    """An artifact is stale, incomplete or belongs to another pipeline."""


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require_equal(actual, expected, context: str) -> None:
    if actual != expected:
        raise ArtifactError(f"{context} mismatch; rebuild the affected artifacts")


def write_manifest(directory, names, filename="artifacts.json") -> None:
    directory = Path(directory)
    files = {name: file_sha256(directory / name) for name in names}
    (directory / filename).write_text(json.dumps({"schema_version": 2, "files": files}, indent=2))


def verify_manifest(directory, names, filename="artifacts.json") -> None:
    directory = Path(directory)
    path = directory / filename
    if not path.is_file():
        raise ArtifactError(f"missing {path.name}; rebuild this legacy or incomplete artifact")
    payload = json.loads(path.read_text())
    require_equal(payload.get("schema_version"), 2, "artifact schema")
    files = payload.get("files", {})
    require_equal(set(files), set(names), "artifact file list")
    for name in names:
        if not (directory / name).is_file():
            raise ArtifactError(f"missing artifact file: {name}")
        require_equal(file_sha256(directory / name), files[name], f"artifact {name}")


def write_embedding_metadata(path, contract: Mapping, shape) -> None:
    path = Path(path)
    payload = {
        "schema_version": 2, "contract": dict(contract),
        "shape": list(shape), "sha256": file_sha256(path),
    }
    temporary = Path(str(path) + ".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(Path(str(path) + ".json"))


def load_embedding_cache(path, contract: Mapping, shape):
    import numpy as np

    path = Path(path)
    sidecar = Path(str(path) + ".json")
    if not sidecar.is_file():
        raise ArtifactError("embedding cache has no provenance; use --rebuild-embeddings")
    metadata = json.loads(sidecar.read_text())
    require_equal(metadata.get("schema_version"), 2, "embedding schema")
    require_equal(metadata.get("contract"), dict(contract), "embedding source/configuration")
    require_equal(metadata.get("shape"), list(shape), "embedding shape")
    require_equal(metadata.get("sha256"), file_sha256(path), "embedding content")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    require_equal(list(values.shape), list(shape), "embedding array shape")
    require_equal(str(values.dtype), "float32", "embedding dtype")
    return values
