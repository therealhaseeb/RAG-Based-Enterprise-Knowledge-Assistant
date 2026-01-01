from typing import List, Dict, Tuple, Optional


def build_prompt(query: str,
                 contexts: List[Dict],
                 system_prompt_template: Optional[str] = None,
                 token_budget: int = 2048) -> Dict:
    """Assemble a prompt for the LLM using `system_prompt_template` and `contexts`.

    - `contexts` is a list of dicts with at least `text` and `source_id`/`id`.
    - `system_prompt_template` is optional system instruction text.
    - `token_budget` is an approximate token budget (we use characters as a rough proxy).

    Returns a dict: {"prompt": str, "used_contexts": List[Dict]}.
    """
    # Very simple token approximation: 1 token ≈ 4 characters
    char_budget = token_budget * 4

    system_part = system_prompt_template.strip() + "\n\n" if system_prompt_template else ""

    used_contexts: List[Dict] = []
    context_parts: List[str] = []
    used_chars = len(system_part) + len(query)

    for c in contexts:
        text = c.get("text", "")
        cid = c.get("id") or c.get("source_id") or ""
        citation = f"[Source: {cid}]\n"
        part = citation + text + "\n\n"
        if used_chars + len(part) > char_budget:
            # stop adding contexts once budget exceeded
            break
        context_parts.append(part)
        used_contexts.append(c)
        used_chars += len(part)

    # Build final prompt
    context_section = "".join(context_parts)
    prompt = f"{system_part}CONTEXT:\n{context_section}USER QUERY:\n{query}\n"

    return {"prompt": prompt, "used_contexts": used_contexts}


__all__ = ["build_prompt"]
