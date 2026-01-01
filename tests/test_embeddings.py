import unittest
from unittest.mock import patch

from app.embeddings import embed_texts


class TestEmbeddings(unittest.TestCase):
    @patch("app.embeddings.load_embedding_model")
    def test_embed_texts_returns_vectors(self, mock_load):
        class FakeModel:
            def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
                return [[0.1, 0.2, 0.3] for _ in texts]

        mock_load.return_value = FakeModel()
        vecs = embed_texts(["hello", "world"], batch_size=2)
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 3)


if __name__ == "__main__":
    unittest.main()
