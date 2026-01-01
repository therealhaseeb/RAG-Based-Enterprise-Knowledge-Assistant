import uuid
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50, source_id: Optional[str] = None) -> List[Dict]:
    """Use LangChain's RecursiveCharacterTextSplitter to create chunks.

    Returns list of dicts: {"source_id", "position", "text"}.
    """
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )

    texts = splitter.split_text(text)
    base_id = source_id or str(uuid.uuid4())
    return [{"source_id": base_id, "position": i, "text": t} for i, t in enumerate(texts)]


if __name__ == "__main__":
    sample = (
        "This is the first sentence. Here is the second sentence which is a bit longer. "
        "Finally a short third sentence to test chunking behaviour.\n\nNew paragraph to check splitting."
    )
    for c in chunk_text(sample, chunk_size=60, overlap=20):
        print(f"[{c['position']}] ({c['source_id']}) {c['text']}")
