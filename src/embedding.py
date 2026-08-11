from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Charge le modèle utilisé pour transformer le texte en vecteurs.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu", "revision": MODEL_REVISION},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embeddings
