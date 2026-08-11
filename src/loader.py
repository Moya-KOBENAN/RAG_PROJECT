from pathlib import Path
import logging

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
LOGGER = logging.getLogger(__name__)


def load_documents(directory: str | Path):
    documents = []
    folder = Path(directory)

    if not folder.exists():
        raise FileNotFoundError(
            f"Le dossier '{directory}' n'existe pas."
        )

    for file_path in sorted(folder.iterdir(), key=lambda path: path.name.lower()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))

            elif suffix == ".txt":
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8",
                )

            elif suffix == ".docx":
                loader = Docx2txtLoader(str(file_path))

            else:
                LOGGER.debug("Format ignoré : %s", file_path.name)
                continue

            loaded_documents = loader.load()

            for document in loaded_documents:
                document.metadata["source_name"] = file_path.name

            documents.extend(loaded_documents)

            LOGGER.info("Document chargé : %s", file_path.name)

        except Exception as error:
            raise RuntimeError(
                f"Impossible de charger '{file_path.name}' : {error}"
            ) from error

    if not documents:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Aucun document exploitable dans '{folder}'. "
            f"Formats acceptés : {supported}."
        )

    return documents
