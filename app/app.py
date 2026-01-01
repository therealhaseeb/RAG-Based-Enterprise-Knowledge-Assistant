"""Top-level app module. Keep lightweight; utilities live under `app.rag`.

This module preserves a backward-compatible `chunk_text` symbol by
re-exporting the implementation from `app.rag.utils`.
"""
from app.rag.utils import chunk_text


__all__ = ["chunk_text"]


if __name__ == "__main__":
	# simple smoke demo
	sample = (
		"This is the first sentence. Here is the second sentence which is a bit longer. "
		"Finally a short third sentence to test chunking behaviour.\n\nNew paragraph to check splitting."
	)
	for c in chunk_text(sample, chunk_size=60, overlap=20):
		print(f"[{c['position']}] ({c['source_id']}) {c['text']}")

