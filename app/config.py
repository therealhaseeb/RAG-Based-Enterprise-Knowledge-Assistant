import os
from dataclasses import dataclass
from typing import Optional

# Optional support for reading a local .env file during development
try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore
except Exception:
    _load_dotenv = None


@dataclass
class Config:
    gemini_api_key: str
    pinecone_api_key: str
    pinecone_environment: Optional[str]
    pinecone_index: str
    hf_embedding_model: str
    hf_api_key: Optional[str]
    pinecone_metric: str


def load_config() -> Config:
    """Load and validate configuration from environment variables.

    This function will attempt to load a local `.env` file (from the current
    working directory) if `python-dotenv` is installed. After that it reads
    environment variables.

    Required env vars:
    - `GEMINI_API_KEY` (for Gemini LLM calls)
    - `PINECONE_API_KEY` (for Pinecone)

    Optional env vars:
    - `PINECONE_ENVIRONMENT` or `PINECONE_ENV`
    - `PINECONE_INDEX` (defaults to `default`)
    - `HF_EMBEDDING_MODEL` (defaults to `sentence-transformers/all-MiniLM-L6-v2`)
    - `HUGGINGFACE_API_KEY`
    - `PINECONE_METRIC`

    Raises `EnvironmentError` if required vars are missing.
    """
    # load .env if available (safe no-op if python-dotenv isn't installed)
    if _load_dotenv is not None:
        try:
            _load_dotenv()
        except Exception:
            # don't fail on .env loading errors; environment variables still may be present
            pass

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    missing = []
    if not gemini_api_key:
        missing.append("GEMINI_API_KEY")
    if not pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV")
    pinecone_index = os.getenv("PINECONE_INDEX", "default")
    hf_embedding_model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    pinecone_metric = os.getenv("PINECONE_METRIC", "cosine")

    return Config(
        gemini_api_key=gemini_api_key,
        pinecone_api_key=pinecone_api_key,
        pinecone_environment=pinecone_environment,
        pinecone_index=pinecone_index,
        hf_embedding_model=hf_embedding_model,
        hf_api_key=hf_api_key,
        pinecone_metric=pinecone_metric,
    )


__all__ = ["Config", "load_config"]
