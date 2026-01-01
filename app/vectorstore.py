import os
import uuid
import logging
from typing import List, Dict, Optional

import pinecone

from app.embeddings import embed_texts, load_embedding_model

logger = logging.getLogger(__name__)


def init_pinecone(index_name: str, dimension: int, metric: str = "cosine") -> pinecone.Index:
    """Initialize Pinecone client and ensure index exists. Returns Index instance.

    Requires `PINECONE_API_KEY` and optionally `PINECONE_ENVIRONMENT` env vars.
    Idempotent: will not recreate an existing index.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY is not set")

    pinecone.init(api_key=api_key, environment=environment)

    existing = pinecone.list_indexes()
    if index_name not in existing:
        logger.info("Creating Pinecone index '%s' (dim=%s, metric=%s)", index_name, dimension, metric)
        pinecone.create_index(name=index_name, dimension=dimension, metric=metric)
    else:
        logger.info("Pinecone index '%s' already exists", index_name)

    return pinecone.Index(index_name)


def upsert_documents(docs: List[Dict], index: pinecone.Index, model: Optional[object] = None, batch_size: int = 100) -> List[str]:
    """Embed and upsert documents into Pinecone.

    `docs` is a list of dicts with keys: `id` (optional), `text`, `metadata` (optional).
    Returns list of upserted ids.
    """
    if not docs:
        return []

    texts = [d["text"] for d in docs]
    model = model or load_embedding_model()
    embeddings = embed_texts(texts, model=model)

    upsert_ids = []
    vectors = []
    for d, emb in zip(docs, embeddings):
        vid = d.get("id") or str(uuid.uuid4())
        meta = d.get("metadata", {})
        vectors.append((vid, emb, meta))
        upsert_ids.append(vid)

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)

    return upsert_ids


def query_pinecone(query: str, index: pinecone.Index, top_k: int = 5, model: Optional[object] = None) -> Dict:
    """Query Pinecone with an embedded query. Returns raw Pinecone response.

    Caller can post-process to extract documents, scores and metadata.
    """
    model = model or load_embedding_model()
    q_emb = embed_texts([query], model=model)[0]
    resp = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    return resp


__all__ = ["init_pinecone", "upsert_documents", "query_pinecone"]
