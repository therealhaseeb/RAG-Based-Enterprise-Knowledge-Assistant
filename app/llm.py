import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def call_gemini(prompt: str, temperature: float = 0.0, stream: bool = False, mock: bool = False) -> str:
    """Call Gemini (Google) generative model.

    - If `mock=True`, returns a canned response useful for tests.
    - Otherwise attempts to use `google.generativeai` (official SDK). If not
      available or `GEMINI_API_KEY` missing, raises an informative error.
    """
    if mock:
        return (
            "MOCK_RESPONSE: This is a mocked Gemini answer.\nSources: [Source: mock-doc-1]"
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set. Set it or call with mock=True for testing.")

    # Prefer official google generative ai package if present
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        # use a reasonable default model name; users can change via env if desired
        model = os.getenv("GEMINI_MODEL", "models/text-bison-001")
        resp = genai.generate_text(model=model, input=prompt)
        # resp may be a complex object; try to extract text
        if hasattr(resp, "text"):
            return resp.text
        # fallback: string representation
        return str(resp)
    except Exception:
        logger.exception("Failed to call google.generativeai; ensure it's installed and GEMINI_API_KEY is set")
        raise


__all__ = ["call_gemini"]
