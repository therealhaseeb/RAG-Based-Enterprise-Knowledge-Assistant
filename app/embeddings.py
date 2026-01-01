import os
import logging
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def load_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """Load a HuggingFace / sentence-transformers model for embeddings.

    - Reads `HF_EMBEDDING_MODEL` env var if `model_name` is not provided.
    - Uses GPU if available.
    """
    model_name = model_name or os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading embedding model '%s' on device %s", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    return model


def embed_texts(texts: List[str], model: Optional[SentenceTransformer] = None, batch_size: int = 32) -> List[List[float]]:
    """Return embeddings for `texts` as a list of float vectors.

    - If `model` is not provided, the default HF model will be loaded.
    - This uses `SentenceTransformer.encode` with batching and returns Python lists.
    """
    if not texts:
        return []

    if model is None:
        model = load_embedding_model()

    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except Exception:
        # Fallback: try without convert_to_numpy
        logger.exception("Embedding call failed with convert_to_numpy, retrying without it")
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)

    # Ensure return type is List[List[float]]
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return [list(map(float, e)) for e in embeddings]


__all__ = ["load_embedding_model", "embed_texts"]
