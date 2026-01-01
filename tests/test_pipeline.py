import unittest
from unittest.mock import MagicMock

from app.rag.pipeline import generate_answer


class TestPipeline(unittest.TestCase):
    def test_generate_answer_with_mocked_components(self):
        # create a fake index returning predictable results
        class FakeIndex:
            def query(self, *args, **kwargs):
                return {"matches": [{"id": "doc1", "score": 0.9, "metadata": {"text": "Doc1 text"}}]}

        fake_index = FakeIndex()

        # mock LLM
        def fake_llm(prompt):
            return "This is a mocked answer."

        out = generate_answer("What is X?", index=fake_index, top_k=1, re_rank=False, call_gemini_fn=fake_llm)
        self.assertIn("answer", out)
        self.assertEqual(out["answer"], "This is a mocked answer.")
        self.assertIn("sources", out)
        self.assertGreaterEqual(len(out["sources"]), 0)


if __name__ == "__main__":
    unittest.main()
