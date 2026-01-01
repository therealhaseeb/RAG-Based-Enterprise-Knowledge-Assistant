import unittest
from unittest.mock import MagicMock

from app.rag.retriever import retrieve_documents


class TestRetriever(unittest.TestCase):
    def test_retrieve_documents_structure(self):
        # Fake Pinecone index with expected response shape
        class FakeIndex:
            def query(self, *args, **kwargs):
                return {"matches": [{"id": "doc1", "score": 0.8, "metadata": {"text": "hello world"}}]}

        idx = FakeIndex()
        out = retrieve_documents("hello", index=idx, top_k=1, re_rank=False)
        self.assertIn("results", out)
        self.assertIsInstance(out["results"], list)


if __name__ == "__main__":
    unittest.main()
