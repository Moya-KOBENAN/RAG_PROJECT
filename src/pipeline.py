import logging
import re
import unicodedata
from dataclasses import dataclass

from src.chunker import split_documents
from src.config import Settings
from src.embedding import get_embedding_model
from src.loader import load_documents
from src.manifest import (
    build_manifest,
    corpus_description,
    index_is_current,
    read_manifest,
)
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    publish_vector_store,
    validate_saved_index,
)


LOGGER = logging.getLogger(__name__)

QUESTION_STOP_WORDS = {
    "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle",
    "en", "est", "et", "il", "la", "le", "les", "leur", "leurs", "ma",
    "mes", "mon", "moyen", "moyenne", "ne", "nos", "notre", "ou", "par", "pas",
    "pour", "prix",
    "que", "quel", "quelle", "quels", "quelles", "qui", "sa", "ses", "son",
    "sur", "ta", "tes", "ton", "un", "une", "variation", "vos", "votre",
}


def _normalized_words(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return set(re.findall(r"[a-z0-9]+", without_accents))


def _question_terms(question: str) -> set[str]:
    return {
        word
        for word in _normalized_words(question)
        if len(word) >= 3 and word not in QUESTION_STOP_WORDS
    }


def _terms_match(question_terms: set[str], document_words: set[str]) -> bool:
    return any(
        question_term == document_word
        or (
            min(len(question_term), len(document_word)) >= 4
            and (
                question_term.startswith(document_word)
                or document_word.startswith(question_term)
            )
        )
        for question_term in question_terms
        for document_word in document_words
    )


@dataclass(frozen=True)
class IngestionResult:
    status: str
    pages: int = 0
    chunks: int = 0


def search_index(settings: Settings, question: str, limit: int = 5) -> list[dict]:
    """Retourne uniquement les passages dont la distance est acceptable."""
    validate_current_index(settings)
    embeddings = get_embedding_model()
    vector_store = load_vector_store(settings.index_directory, embeddings)
    # On récupère davantage de candidats avant le filtrage lexical afin qu'un
    # voisin hors sujet ne masque pas un passage qui contient réellement le terme.
    normalized_question = question.strip().casefold()
    results = vector_store.similarity_search_with_score(
        normalized_question,
        k=limit * 5,
    )
    question_terms = _question_terms(question)

    exact_results = []
    approximate_results = []
    for document, score in results:
        if float(score) > settings.max_search_distance:
            continue
        document_words = _normalized_words(document.page_content)
        if question_terms and not _terms_match(question_terms, document_words):
            continue
        item = {
            "text": document.page_content,
            "source": document.metadata.get("source_name"),
            "page": document.metadata.get("page"),
        }
        if not question_terms or question_terms.intersection(document_words):
            exact_results.append(item)
        else:
            approximate_results.append(item)

    relevant_results = exact_results or approximate_results
    return relevant_results[:limit]


def validate_current_index(settings: Settings) -> dict:
    files = corpus_description(settings.documents_directory)
    if not index_is_current(
        settings.index_directory,
        files,
        settings.chunk_size,
        settings.chunk_overlap,
    ):
        raise RuntimeError("L'index FAISS est absent, incomplet ou obsolète.")

    manifest = read_manifest(settings.index_directory)
    expected_vectors = int(manifest["results"]["chunks"])
    validate_saved_index(settings.index_directory, expected_vectors)
    LOGGER.info("Index courant validé : %s vecteurs.", expected_vectors)
    return manifest


def run_ingestion(settings: Settings, force: bool = False) -> IngestionResult:
    files = corpus_description(settings.documents_directory)
    if not force and index_is_current(
        settings.index_directory,
        files,
        settings.chunk_size,
        settings.chunk_overlap,
    ):
        LOGGER.info("Corpus inchangé : l'index existant est conservé.")
        return IngestionResult(status="unchanged")

    LOGGER.info("Chargement de %s document(s).", len(files))
    documents = load_documents(settings.documents_directory)
    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embeddings = get_embedding_model()
    vector_store = create_vector_store(chunks, embeddings)
    manifest = build_manifest(
        files,
        settings.chunk_size,
        settings.chunk_overlap,
        page_count=len(documents),
        chunk_count=len(chunks),
    )
    publish_vector_store(vector_store, settings.index_directory, manifest)
    return IngestionResult(status="rebuilt", pages=len(documents), chunks=len(chunks))
