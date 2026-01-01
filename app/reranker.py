import json
import logging
import re
from typing import List, Dict, Callable, Optional

logger = logging.getLogger(__name__)


def _simple_lexical_score(query: str, text: str) -> int:
    q_tokens = set(re.findall(r"\w+", query.lower()))
    t_tokens = set(re.findall(r"\w+", text.lower()))
    return len(q_tokens & t_tokens)


def rerank_documents_with_gemini(
    query: str,
    docs: List[Dict],
    call_gemini: Optional[Callable[[str], str]] = None,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """Rerank `docs` for `query` using Gemini if provided, otherwise fallback.

    - `docs` should be a list of dicts containing at least `text` and an `id` or `source_id`.
    - If `call_gemini` is provided, it will be invoked with a prompt asking for a
      JSON array of `{id, score, rationale}`. The function will parse and apply
      those scores to the documents.
    - Returns the documents annotated with `rerank_score` and `rerank_rationale`, sorted descending.
    """
    if not docs:
        return []

    if call_gemini:
        prompt_lines = [
            "You are a relevance scorer. Given a user query, score each document between 0 and 1 and include a brief rationale.",
            "Return a JSON array of objects with fields: id, score (0-1), rationale. Respond with JSON only.",
        ]
        prompt_lines.append(f"Query: {query}")
        prompt_lines.append("Documents:")
        for d in docs:
            did = d.get("id") or d.get("source_id") or ""
            text = d.get("text", "")
            prompt_lines.append(f"ID: {did}\nText: {text}\n---")

        prompt = "\n".join(prompt_lines)
        try:
            resp = call_gemini(prompt)
            parsed = json.loads(resp)
            id_map = {item.get("id"): item for item in parsed if item.get("id") is not None}
            out = []
            for d in docs:
                did = d.get("id") or d.get("source_id") or ""
                item = id_map.get(did)
                if item:
                    score = float(item.get("score", 0.0))
                    rationale = item.get("rationale", "")
                else:
                    score = 0.0
                    rationale = ""
                new = dict(d)
                new["rerank_score"] = score
                new["rerank_rationale"] = rationale
                out.append(new)
            out.sort(key=lambda x: x["rerank_score"], reverse=True)
            if top_k:
                out = out[:top_k]
            return out
        except Exception:
            logger.exception("Gemini rerank failed, falling back to lexical scorer")

    # Fallback lexical scorer
    scored = []
    for d in docs:
        text = d.get("text", "")
        score = _simple_lexical_score(query, text)
        new = dict(d)
        new["rerank_score"] = float(score)
        new["rerank_rationale"] = f"lexical_overlap={score}"
        scored.append(new)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    if top_k:
        scored = scored[:top_k]
    return scored


__all__ = ["rerank_documents_with_gemini"]
