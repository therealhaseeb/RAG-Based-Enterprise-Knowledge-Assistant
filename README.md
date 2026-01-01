# RAG-Based Enterprise Knowledge Assistant

Minimal scaffold for a Retrieval-Augmented Generation (RAG) assistant.

Features:
- Ingest PDFs / text / webpages
- Create embeddings (OpenAI by default)
- Store embeddings in Pinecone
- Retrieve relevant chunks and answer via an LLM (OpenAI by default)

Tech stack:
- LLM: configurable (default: OpenAI)
- Framework: LangChain
- Vector DB: Pinecone
- Backend: FastAPI
- UI: Streamlit

Quickstart

1. Copy `.env.example` to `.env` and set keys (`OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_ENV`, `PINECONE_INDEX`).
2. Install deps:

```
pip install -r requirements.txt
```

3. Run backend:

```
uvicorn app.backend.main:app --reload
```

4. Run UI:

```
streamlit run app/frontend/streamlit_app.py
```

Notes
- The scaffold uses OpenAI for embeddings and the chat model by default. To use Gemini (PaLM/Vertex AI) with LangChain, add or customize the LLM wrapper in `app/backend/llm.py` and set `LLM_PROVIDER=gemini` in `.env`.
# RAG-Based-Enterprise-Knowledge-Assistant