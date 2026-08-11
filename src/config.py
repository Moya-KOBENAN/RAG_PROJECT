import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _positive_integer(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} doit être strictement positif.")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} doit être strictement positif.")
    return value


@dataclass(frozen=True)
class Settings:
    documents_directory: Path
    index_directory: Path
    chunk_size: int = 500
    chunk_overlap: int = 100
    max_search_distance: float = 2.0

    def __post_init__(self):
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "RAG_CHUNK_OVERLAP doit être positif ou nul et inférieur "
                "à RAG_CHUNK_SIZE."
            )

    @classmethod
    def from_environment(cls):
        return cls(
            documents_directory=Path(
                os.getenv("RAG_DOCUMENTS_DIR", PROJECT_ROOT / "data" / "documents")
            ).expanduser().resolve(),
            index_directory=Path(
                os.getenv("RAG_INDEX_DIR", PROJECT_ROOT / "index" / "faiss_index")
            ).expanduser().resolve(),
            chunk_size=_positive_integer("RAG_CHUNK_SIZE", 500),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
            max_search_distance=_positive_float("RAG_MAX_SEARCH_DISTANCE", 2.0),
        )
