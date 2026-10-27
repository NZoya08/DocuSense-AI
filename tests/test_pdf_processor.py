import unittest

from chunker import build_chunks
from config import MAX_UPLOAD_MB
from pdf_helpers import make_pdf
from pdf_processor import NO_TEXT_MESSAGE, DocumentError, clean_text, extract_document, validate_upload


def numbered_words(prefix, count=40):
    return " ".join(f"{prefix}{i}" for i in range(count))


class ValidationTests(unittest.TestCase):
    def test_rejects_files_without_a_pdf_extension(self):
        with self.assertRaisesRegex(DocumentError, r"\.pdf extension"):
            validate_upload("notes.txt", b"%PDF-1.4 example")

    def test_rejects_empty_files(self):
        with self.assertRaisesRegex(DocumentError, "empty"):
            validate_upload("empty.pdf", b"")

    def test_rejects_files_without_the_pdf_header(self):
        with self.assertRaisesRegex(DocumentError, "does not look like a PDF"):
            validate_upload("fake.pdf", b"hello world, this is plain text")

    def test_rejects_files_over_the_size_limit(self):
        too_large = b"%PDF-" + b"0" * (MAX_UPLOAD_MB * 1024 * 1024)
        with self.assertRaisesRegex(DocumentError, "larger than"):
            validate_upload("big.pdf", too_large)

    def test_accepts_a_plausible_pdf_header(self):
        validate_upload("ok.pdf", b"%PDF-1.7\n% minimal example")  # raises nothing

    def test_clean_text_removes_control_characters_and_extra_spaces(self):
        self.assertEqual(clean_text("Hello\x00   world\r\n\r\n\r\nNext"), "Hello world\n\nNext")


class ExtractionTests(unittest.TestCase):
    def test_extracts_every_page_with_its_page_number(self):
        data = make_pdf([numbered_words("alpha"), numbered_words("bravo"), numbered_words("charlie")])
        document = extract_document("handbook.pdf", data)
        self.assertEqual(document.page_count, 3)
        self.assertEqual([page.page_number for page in document.pages], [1, 2, 3])
        self.assertEqual(document.skipped_pages, [])
        self.assertFalse(document.is_partial)
        self.assertIn("bravo5", document.pages[1].text)

    def test_pages_without_text_are_reported_as_skipped(self):
        data = make_pdf([numbered_words("alpha"), None, numbered_words("charlie")])
        document = extract_document("mixed.pdf", data)
        self.assertEqual(document.skipped_pages, [2])
        self.assertTrue(document.is_partial)
        self.assertEqual([page.page_number for page in document.pages], [1, 3])

    def test_pdf_without_selectable_text_gets_the_ocr_message(self):
        with self.assertRaises(DocumentError) as context:
            extract_document("scan.pdf", make_pdf([None, None]))
        self.assertEqual(str(context.exception), NO_TEXT_MESSAGE)

    def test_pdf_with_only_a_few_characters_is_rejected(self):
        with self.assertRaisesRegex(DocumentError, "characters of selectable text"):
            extract_document("tiny.pdf", make_pdf(["Hi"]))

    def test_damaged_pdf_raises_a_friendly_error(self):
        with self.assertRaises(DocumentError):
            extract_document("broken.pdf", b"%PDF-1.4\n" + b"garbage " * 200)

    def test_chunk_page_labels_match_the_page_the_text_came_from(self):
        data = make_pdf([numbered_words("alpha"), numbered_words("bravo")])
        document = extract_document("handbook.pdf", data)
        chunks = build_chunks(document.pages, document.name, chunk_words=40, overlap_words=0)
        self.assertEqual([chunk.page_label for chunk in chunks], ["Page 1", "Page 2"])
        self.assertTrue(chunks[1].text.startswith("bravo0"))


if __name__ == "__main__":
    unittest.main()
