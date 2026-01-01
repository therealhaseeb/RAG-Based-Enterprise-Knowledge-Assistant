from typing import List, Dict, Optional

from app.vectorstore import query_pinecone
from app.reranker import rerank_documents_with_gemini


def retrieve_documents(
    query: str,
    index,
    top_k: int = 5,
    re_rank: bool = True,
    call_gemini: Optional[callable] = None,
) -> Dict:
    """Retrieve documents for `query` from Pinecone index and optionally rerank.

    - `index` is a Pinecone Index instance (returned by `init_pinecone`).
    - If `re_rank` is True, documents will be passed to `rerank_documents_with_gemini`.
    - `call_gemini` is an optional callable(prompt) -> str used by reranker.

    Returns a dict with keys: `query`, `results` (list of docs), `raw_response`.
    Each result contains: `id`, `text`, `score`, `metadata`, and optional `rerank_score`.
    """
    if index is None:
        raise ValueError("`index` (Pinecone Index) is required")

    resp = query_pinecone(query, index=index, top_k=top_k)

    matches = resp.get("matches") or resp.get("results") or []

    docs: List[Dict] = []
    for m in matches:
        mid = m.get("id") or m.get("metadata", {}).get("id")
        score = m.get("score") or m.get("score", 0.0)
        metadata = m.get("metadata", {})
        # prefer textual fields in metadata
        text = metadata.get("text") or metadata.get("content") or metadata.get("page_content") or ""
        docs.append({"id": mid, "text": text, "score": score, "metadata": metadata})

    if re_rank:
        # reranker will fallback to lexical scoring if call_gemini is None
        docs = rerank_documents_with_gemini(query, docs, call_gemini=call_gemini, top_k=top_k)

    return {"query": query, "results": docs, "raw_response": resp}


__all__ = ["retrieve_documents"]
