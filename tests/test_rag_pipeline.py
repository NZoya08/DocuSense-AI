import unittest

from chunker import Chunk, PageText, build_chunks
from llm_service import Answer, LLMServiceError
from pdf_helpers import make_pdf
from pdf_processor import ExtractedDocument, extract_document
from rag_pipeline import (
    NOT_FOUND_MESSAGE,
    SummaryResult,
    answer_question,
    build_summary_file,
    format_page_list,
    safe_file_stem,
    summarize_document,
    validate_question,
)
from retriever import DocumentIndex, KeywordEmbedder


class FakeService:
    def __init__(self, answer=None, error=None):
        self.answer_value = answer
        self.error = error
        self.answer_calls = 0
        self.summary_inputs = []

    def answer(self, question, passages):
        self.answer_calls += 1
        if self.error is not None:
            raise self.error
        return self.answer_value

    def summarize(self, doc_name, document_text):
        self.summary_inputs.append(document_text)
        return "## Main purpose\nA leave policy."


def index_for(chunks):
    index = DocumentIndex(KeywordEmbedder())
    index.build(chunks)
    return index


def handbook_chunks():
    return [
        Chunk("handbook.pdf", "Full-time employees receive 20 days of paid annual leave each year.", 1, 1),
        Chunk("handbook.pdf", "Travel claims must be submitted within 30 days of the trip.", 2, 2),
    ]


class QuestionAnsweringTests(unittest.TestCase):
    def test_supported_question_returns_the_cited_passage_and_its_page(self):
        service = FakeService(answer=Answer("Staff get 20 days of leave [1].", False, (1,)))
        result = answer_question(index_for(handbook_chunks()), service, "How many days of annual leave do employees get?")
        self.assertTrue(result.answered)
        self.assertTrue(result.cited)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].page_label, "Page 1")
        self.assertEqual(service.answer_calls, 1)

    def test_question_without_matching_evidence_never_reaches_the_model(self):
        service = FakeService(answer=Answer("should not be used", False, ()))
        result = answer_question(index_for(handbook_chunks()), service, "What is the chief executive's favourite colour?")
        self.assertFalse(result.answered)
        self.assertEqual(result.message, NOT_FOUND_MESSAGE)
        self.assertEqual(service.answer_calls, 0)

    def test_model_reported_insufficient_evidence_is_not_shown_as_an_answer(self):
        service = FakeService(answer=Answer(text="", insufficient_evidence=True))
        result = answer_question(index_for(handbook_chunks()), service, "How many annual leave days are there?")
        self.assertFalse(result.answered)
        self.assertEqual(result.message, NOT_FOUND_MESSAGE)

    def test_uncited_answer_is_flagged_and_all_passages_are_shown(self):
        service = FakeService(answer=Answer(text="Staff get annual leave.", insufficient_evidence=False))
        result = answer_question(index_for(handbook_chunks()), service, "annual leave for staff")
        self.assertTrue(result.answered)
        self.assertFalse(result.cited)
        self.assertGreaterEqual(len(result.sources), 1)

    def test_api_errors_reach_the_caller_as_llm_service_errors(self):
        service = FakeService(error=LLMServiceError("Gemini is busy"))
        with self.assertRaises(LLMServiceError):
            answer_question(index_for(handbook_chunks()), service, "annual leave days")

    def test_empty_question_is_rejected_before_any_work(self):
        service = FakeService(answer=Answer("x", False, ()))
        with self.assertRaises(ValueError):
            answer_question(index_for(handbook_chunks()), service, "   ")
        self.assertEqual(service.answer_calls, 0)

    def test_overlong_question_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_question("x" * 600)

    def test_end_to_end_from_pdf_bytes_keeps_the_page_number(self):
        page_one = " ".join(["annual", "leave", "is", "20", "days"] + [f"filler{i}" for i in range(35)])
        page_two = " ".join(["travel", "claims", "within", "30", "days"] + [f"note{i}" for i in range(35)])
        document = extract_document("handbook.pdf", make_pdf([page_one, page_two]))
        chunks = build_chunks(document.pages, document.name, chunk_words=40, overlap_words=0)
        service = FakeService(answer=Answer("Annual leave is 20 days [1].", False, (1,)))
        result = answer_question(index_for(chunks), service, "How many annual leave days are there?")
        self.assertEqual(result.sources[0].page_label, "Page 1")
        self.assertIn("annual leave", result.sources[0].text)


class SummaryTests(unittest.TestCase):
    def test_summary_reports_skipped_pages_and_truncation(self):
        document = ExtractedDocument(
            name="report.pdf",
            page_count=4,
            pages=[PageText(1, "a" * 100), PageText(2, "b" * 100), PageText(4, "c" * 100)],
            skipped_pages=[3],
        )
        service = FakeService()
        summary = summarize_document(service, document, max_chars=150)
        self.assertEqual(summary.included_pages, (1,))
        self.assertEqual(len(summary.incomplete_notes), 2)
        self.assertIn("pages 3", summary.incomplete_notes[0])
        self.assertIn("1 of the 3 pages", summary.incomplete_notes[1])
        self.assertIn("[Page 1]", service.summary_inputs[0])
        self.assertNotIn("[Page 2]", service.summary_inputs[0])

    def test_complete_document_has_no_notes(self):
        document = ExtractedDocument(
            name="short.pdf",
            page_count=1,
            pages=[PageText(1, "Some text here.")],
            skipped_pages=[],
        )
        summary = summarize_document(FakeService(), document)
        self.assertEqual(summary.incomplete_notes, [])

    def test_summary_file_contains_the_notes_and_the_disclaimer(self):
        summary = SummaryResult(
            markdown="## Conclusion\nDone.",
            included_pages=(1,),
            page_count=2,
            text_page_count=1,
            skipped_pages=(2,),
        )
        content = build_summary_file("report.pdf", summary)
        self.assertTrue(content.startswith("# Summary of report.pdf"))
        self.assertIn("> Note:", content)
        self.assertIn("Check important dates", content)

    def test_download_file_names_are_made_safe(self):
        self.assertEqual(safe_file_stem("My Report (final).pdf"), "My_Report_final")
        self.assertEqual(safe_file_stem("..."), "document")

    def test_page_lists_are_formatted_and_shortened(self):
        self.assertEqual(format_page_list([2, 5, 9]), "2, 5, 9")
        self.assertTrue(format_page_list(list(range(1, 30))).endswith(", ..."))


if __name__ == "__main__":
    unittest.main()
