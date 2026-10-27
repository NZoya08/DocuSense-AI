import importlib.util
import unittest
from unittest import mock

from chunker import Chunk
from retriever import DocumentIndex, KeywordEmbedder, SemanticEmbedder, create_embedder


def sample_chunks():
    return [
        Chunk("handbook.pdf", "Full-time employees receive 20 days of paid annual leave each calendar year.", 1, 1),
        Chunk(
            "handbook.pdf",
            "Business travel expenses are reimbursed when claims are submitted within 30 days of the trip.",
            2,
            2,
        ),
        Chunk(
            "handbook.pdf",
            "The annual performance review window opens on 1 March 2026 and closes on 31 March 2026.",
            3,
            3,
        ),
    ]


class KeywordRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = DocumentIndex(KeywordEmbedder())
        self.index.build(sample_chunks())

    def test_question_is_matched_to_the_passage_that_shares_its_words(self):
        results = self.index.search("How many days of annual leave do employees get?", top_k=3, min_score=0.05)
        self.assertEqual(results[0].chunk.page_label, "Page 1")
        self.assertGreater(results[0].score, 0.05)

    def test_simple_plurals_match(self):
        results = self.index.search("When are travel expense claims reimbursed?", top_k=1)
        self.assertEqual(results[0].chunk.page_label, "Page 2")

    def test_question_with_no_shared_words_returns_no_passages(self):
        results = self.index.search("What is the chief executive's favourite colour?", top_k=3, min_score=0.05)
        self.assertEqual(results, [])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.search("   ")

    def test_empty_chunk_list_is_rejected(self):
        with self.assertRaises(ValueError):
            DocumentIndex(KeywordEmbedder()).build([])

    def test_searching_before_building_is_rejected(self):
        with self.assertRaises(RuntimeError):
            DocumentIndex(KeywordEmbedder()).search("leave")


class EmbedderSelectionTests(unittest.TestCase):
    def test_keyword_backend_needs_no_model(self):
        embedder, notice = create_embedder("keyword")
        self.assertEqual(embedder.kind, "keyword")
        self.assertIsNone(notice)

    def test_failure_to_load_the_semantic_model_falls_back_with_a_notice(self):
        with mock.patch("retriever.importlib.util.find_spec", return_value=object()):
            with mock.patch("retriever.SemanticEmbedder", side_effect=OSError("model download blocked")):
                embedder, notice = create_embedder("semantic")
        self.assertEqual(embedder.kind, "keyword")
        self.assertIn("keyword matching", notice)

    def test_missing_package_falls_back_with_a_notice(self):
        with mock.patch("retriever.importlib.util.find_spec", return_value=None):
            embedder, notice = create_embedder("semantic")
        self.assertEqual(embedder.kind, "keyword")
        self.assertIn("not installed", notice)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            create_embedder("magic")


@unittest.skipUnless(
    importlib.util.find_spec("sentence_transformers"),
    "sentence-transformers is not installed, so the semantic test is skipped",
)
class SemanticRetrievalTests(unittest.TestCase):
    def test_paraphrased_question_finds_the_leave_passage(self):
        index = DocumentIndex(SemanticEmbedder())
        index.build(sample_chunks())
        results = index.search("How many paid vacation days do staff receive each year?", top_k=1)
        self.assertEqual(results[0].chunk.page_label, "Page 1")


if __name__ == "__main__":
    unittest.main()
