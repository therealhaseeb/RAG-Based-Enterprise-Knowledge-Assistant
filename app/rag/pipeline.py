from typing import Optional

from app.rag.retriever import retrieve_documents
from app.rag.prompt import build_prompt
from app.llm import call_gemini


def generate_answer(
    query: str,
    index,
    top_k: int = 5,
    re_rank: bool = True,
    call_gemini_fn=None,
    system_prompt: Optional[str] = None,
):
    """End-to-end RAG pipeline: retrieve -> prompt -> LLM -> post-process.

    - `index` is required (Pinecone Index or compatible mock).
    - `call_gemini_fn` can be provided to override the LLM call (useful in tests).
    Returns dict: {answer: str, sources: List[dict], retrieval: dict}
    """
    # 1) Retrieve
    retrieval = retrieve_documents(query, index=index, top_k=top_k, re_rank=re_rank, call_gemini=call_gemini_fn)
    contexts = retrieval.get("results", [])

    # 2) Build prompt
    prompt_bundle = build_prompt(query, contexts, system_prompt_template=system_prompt)
    prompt = prompt_bundle["prompt"]

    # 3) Call LLM
    llm_caller = call_gemini_fn if call_gemini_fn is not None else call_gemini
    resp = llm_caller(prompt)

    # 4) Post-process: try to extract cited sources from contexts
    sources = []
    for c in prompt_bundle.get("used_contexts", []):
        sources.append({"id": c.get("id") or c.get("source_id"), "metadata": c.get("metadata", {})})

    return {"answer": resp, "sources": sources, "retrieval": retrieval}


__all__ = ["generate_answer"]
