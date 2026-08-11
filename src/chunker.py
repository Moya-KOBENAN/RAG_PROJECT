import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter


LOGGER = logging.getLogger(__name__)


def split_documents(
    documents,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
):
    """
    Découpe les documents en petits morceaux appelés chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size doit être strictement positif.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap doit être positif ou nul et inférieur à chunk_size."
        )

    if not documents:
        raise ValueError("Aucun document disponible pour le découpage.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    LOGGER.info("Nombre total de chunks créés : %s", len(chunks))

    return chunks
