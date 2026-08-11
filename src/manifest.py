import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import faiss

from src.embedding import MODEL_NAME, MODEL_REVISION
from src.loader import SUPPORTED_EXTENSIONS


MANIFEST_FILENAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_description(directory: Path) -> list[dict]:
    if not directory.exists():
        raise FileNotFoundError(f"Le dossier '{directory}' n'existe pas.")

    files = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(
                {"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
            )
    if not files:
        raise ValueError(f"Aucun document exploitable dans '{directory}'.")
    return files


def ingestion_signature(files: list[dict], chunk_size: int, chunk_overlap: int) -> str:
    reproducibility_data = {
        "documents": files,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "normalize_embeddings": True,
    }
    serialized = json.dumps(reproducibility_data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_manifest(files, chunk_size, chunk_overlap, page_count, chunk_count):
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signature": ingestion_signature(files, chunk_size, chunk_overlap),
        "documents": files,
        "parameters": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "normalize_embeddings": True,
        },
        "results": {"pages": page_count, "chunks": chunk_count},
    }


def read_manifest(index_directory: Path) -> dict | None:
    path = index_directory / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def index_is_current(index_directory, files, chunk_size, chunk_overlap) -> bool:
    index_directory = Path(index_directory)
    manifest = read_manifest(index_directory)
    expected = ingestion_signature(files, chunk_size, chunk_overlap)
    if not manifest or manifest.get("signature") != expected:
        return False

    faiss_path = index_directory / "index.faiss"
    metadata_path = index_directory / "index.pkl"
    if not faiss_path.is_file() or not metadata_path.is_file():
        return False

    try:
        expected_vectors = int(manifest["results"]["chunks"])
        return faiss.read_index(str(faiss_path)).ntotal == expected_vectors
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
