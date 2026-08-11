import logging
import threading

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.config import Settings
from src.pipeline import run_ingestion, search_index, validate_current_index
from src.price_extractor import extract_answer


app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="API de pilotage du pipeline RAG",
)

LOGGER = logging.getLogger(__name__)
INGESTION_LOCK = threading.Lock()


class SearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "RAG API is running",
    }


@app.get("/health")
@app.get("/health/live")
def liveness():
    return {
        "status": "healthy",
    }


@app.get("/health/ready")
def readiness():
    try:
        manifest = validate_current_index(Settings.from_environment())
        return {
            "status": "ready",
            "vectors": manifest["results"]["chunks"],
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="L'index RAG n'est pas prêt.",
        )


@app.get("/index/status")
def index_status():
    try:
        settings = Settings.from_environment()

        manifest = validate_current_index(settings)

        return {
            "status": "valid",
            "manifest": manifest,
        }

    except Exception:
        LOGGER.exception("Impossible de valider l'index RAG.")
        raise HTTPException(
            status_code=500,
            detail="Impossible de valider l'index RAG.",
        )


@app.post("/ingestion/run")
def ingestion_run(force: bool = False):
    if not INGESTION_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Une ingestion est déjà en cours.",
        )

    try:
        settings = Settings.from_environment()

        result = run_ingestion(
            settings=settings,
            force=force,
        )

        return {
            "status": result.status,
            "pages": result.pages,
            "chunks": result.chunks,
        }

    except Exception:
        LOGGER.exception("Échec de l'ingestion RAG.")
        raise HTTPException(
            status_code=500,
            detail="Échec de l'ingestion RAG.",
        )
    finally:
        INGESTION_LOCK.release()


@app.post("/query")
def query(request: SearchRequest):
    """Recherche les passages pertinents dans les documents indexés."""
    try:
        results = search_index(
            Settings.from_environment(),
            question=request.question,
            limit=3,
        )
        answer = extract_answer(request.question, results)
        return {
            "question": request.question,
            "answer": (
                answer
                if results
                else "Aucune information pertinente n'a été trouvée dans les documents disponibles."
            ),
            "results": results,
        }
    except Exception:
        LOGGER.exception("Échec de la recherche dans l'index RAG.")
        raise HTTPException(
            status_code=500,
            detail="Échec de la recherche dans l'index RAG.",
        )
