import json
import logging
import os
import shutil
import uuid
from pathlib import Path

import faiss
from langchain_community.vectorstores import FAISS


LOGGER = logging.getLogger(__name__)


def load_vector_store(directory: str | Path, embeddings):
    """Charge un index FAISS local validé par l'application."""
    directory = Path(directory)
    if not (directory / "index.faiss").is_file() or not (
        directory / "index.pkl"
    ).is_file():
        raise RuntimeError("L'index FAISS est absent ou incomplet.")

    return FAISS.load_local(
        str(directory),
        embeddings,
        allow_dangerous_deserialization=True,
    )

def create_vector_store(chunks, embeddings):
    """
    Crée une base vectorielle FAISS à partir des chunks.
    """

    if not chunks:
        raise ValueError(
            "Aucun chunk disponible pour créer la base vectorielle."
        )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    LOGGER.info("Base vectorielle FAISS créée avec succès.")

    return vector_store


def save_vector_store(
    vector_store,
    directory: str = "index/faiss_index",
):
    """
    Sauvegarde l'index FAISS sur le disque.
    """

    output_directory = Path(directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(str(output_directory))

    LOGGER.info("Index FAISS sauvegardé dans : %s", output_directory)


def validate_saved_index(directory: str | Path, expected_vectors: int) -> None:
    directory = Path(directory)
    faiss_path = directory / "index.faiss"
    metadata_path = directory / "index.pkl"
    if not faiss_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("L'index sauvegardé est incomplet.")

    index = faiss.read_index(str(faiss_path))
    if index.ntotal != expected_vectors:
        raise RuntimeError(
            f"Index invalide : {index.ntotal} vecteurs au lieu de {expected_vectors}."
        )


def publish_vector_store(vector_store, directory: str | Path, manifest: dict) -> None:
    """Valide puis publie l'index sans exposer un résultat partiel."""
    target = Path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"

    try:
        save_vector_store(vector_store, staging)
        expected_vectors = int(manifest["results"]["chunks"])
        validate_saved_index(staging, expected_vectors)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        had_previous_index = target.exists()
        if had_previous_index:
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except Exception:
            if had_previous_index and backup.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        LOGGER.info("Index validé et publié : %s", target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
