import unittest

from chunker import PageText, build_chunks, build_summary_source


class ChunkingTests(unittest.TestCase):
    def test_chunks_keep_the_page_each_word_came_from(self):
        pages = [PageText(1, "one two three four five"), PageText(2, "six seven eight nine ten")]
        chunks = build_chunks(pages, doc_name="handbook.pdf", chunk_words=5, overlap_words=0)
        self.assertEqual(len(chunks), 2)
        self.assertEqual((chunks[0].page_start, chunks[0].page_end), (1, 1))
        self.assertEqual((chunks[1].page_start, chunks[1].page_end), (2, 2))
        self.assertEqual(chunks[1].text, "six seven eight nine ten")
        self.assertEqual(chunks[1].doc_name, "handbook.pdf")
        self.assertEqual(chunks[1].page_label, "Page 2")

    def test_chunks_never_cross_a_page_boundary(self):
        pages = [PageText(1, "a b c"), PageText(2, "d e f")]
        chunks = build_chunks(pages, "handbook.pdf", chunk_words=4, overlap_words=0)
        self.assertEqual([chunk.page_label for chunk in chunks], ["Page 1", "Page 2"])
        self.assertEqual([chunk.text for chunk in chunks], ["a b c", "d e f"])

    def test_neighbouring_chunks_share_the_overlap_words(self):
        pages = [PageText(1, " ".join(f"w{i}" for i in range(1, 11)))]  # w1 ... w10
        chunks = build_chunks(pages, "handbook.pdf", chunk_words=4, overlap_words=2)
        self.assertEqual(chunks[0].text, "w1 w2 w3 w4")
        self.assertEqual(chunks[1].text, "w3 w4 w5 w6")
        self.assertTrue(chunks[-1].text.endswith("w10"))

    def test_invalid_chunk_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            build_chunks([PageText(1, "a b")], "x.pdf", chunk_words=3, overlap_words=3)
        with self.assertRaises(ValueError):
            build_chunks([PageText(1, "a b")], "x.pdf", chunk_words=0, overlap_words=0)

    def test_no_pages_means_no_chunks(self):
        self.assertEqual(build_chunks([], "x.pdf"), [])


class SummarySourceTests(unittest.TestCase):
    def test_summary_source_stops_at_the_character_budget(self):
        pages = [PageText(1, "a" * 50), PageText(2, "b" * 50), PageText(3, "c" * 50)]
        text, included = build_summary_source(pages, max_chars=130)
        self.assertEqual(included, [1, 2])
        self.assertIn("[Page 1]", text)
        self.assertNotIn("c" * 10, text)


if __name__ == "__main__":
    unittest.main()
