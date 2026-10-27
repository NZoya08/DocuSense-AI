"""Checks uploaded PDFs and extracts their text page by page with PyMuPDF (imported as fitz)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from chunker import PageText
from config import MAX_PAGES, MAX_UPLOAD_MB, MIN_TEXT_CHARS

NO_TEXT_MESSAGE = (
    "No selectable text was found in this PDF. It is probably a scanned document made of images. "
    "OCR (reading text from images) is not supported in this version. Try a text-based PDF instead."
)


class DocumentError(Exception):
    """A problem with the uploaded file. The message is written for the person using the app."""


@dataclass
class ExtractedDocument:
    """The text of an uploaded PDF. Only pages with usable text appear in pages."""

    name: str
    page_count: int
    pages: list[PageText] = field(default_factory=list)
    skipped_pages: list[int] = field(default_factory=list)  # 1-based numbers of pages without text

    @property
    def text_page_count(self) -> int:
        return len(self.pages)

    @property
    def is_partial(self) -> bool:
        return bool(self.skipped_pages)

    @property
    def status_text(self) -> str:
        if self.is_partial:
            return (
                f"Partial extraction: {len(self.skipped_pages)} of {self.page_count} "
                "pages had no selectable text"
            )
        return "Complete: text was extracted from every page"


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def clean_text(raw: str) -> str:
    """Remove control characters, collapse repeated spaces and tidy blank lines."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def validate_upload(file_name: str, data: bytes) -> None:
    """Raise DocumentError when the upload cannot be a usable PDF."""
    if not file_name.lower().endswith(".pdf"):
        raise DocumentError("Please upload a file with a .pdf extension.")
    if not data:
        raise DocumentError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise DocumentError(f"The file is larger than {MAX_UPLOAD_MB} MB. Upload a smaller PDF.")
    if b"%PDF-" not in data[:1024]:
        raise DocumentError("This file does not look like a PDF because its PDF header is missing.")


def extract_document(file_name: str, data: bytes) -> ExtractedDocument:
    """Validate the upload and return its text page by page. Raises DocumentError on failure."""
    validate_upload(file_name, data)
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises different exceptions for damaged files
        raise DocumentError("The PDF could not be opened. The file may be damaged.") from exc

    try:
        if document.needs_pass:
            raise DocumentError("This PDF is password-protected. Remove the password and upload it again.")
        page_count = document.page_count
        if page_count == 0:
            raise DocumentError("This PDF has no pages.")
        if page_count > MAX_PAGES:
            raise DocumentError(f"This PDF has {page_count} pages. The limit is {MAX_PAGES} pages.")

        pages: list[PageText] = []
        skipped: list[int] = []
        for index in range(page_count):
            page_number = index + 1
            try:
                raw_text = document.load_page(index).get_text("text")
            except Exception:  # an unreadable page is treated like a page without text
                skipped.append(page_number)
                continue
            text = clean_text(raw_text)
            if text:
                pages.append(PageText(page_number=page_number, text=text))
            else:
                skipped.append(page_number)
    finally:
        document.close()

    if not pages:
        raise DocumentError(NO_TEXT_MESSAGE)
    total_chars = sum(len(page.text) for page in pages)
    if total_chars < MIN_TEXT_CHARS:
        raise DocumentError(
            f"Only {total_chars} characters of selectable text were found, which is too little to answer "
            "questions. The PDF may be mostly images."
        )
    return ExtractedDocument(name=file_name, page_count=page_count, pages=pages, skipped_pages=skipped)
