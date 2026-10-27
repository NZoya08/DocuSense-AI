"""Splits page text into overlapping chunks that remember which page they came from.

Chunks never cross a page boundary, so every citation points to exactly one page.
This module has no third-party dependencies, which keeps it easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PageText:
    """Cleaned text from one page. page_number is 1-based, the numbering PDF viewers use."""

    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A passage of the document plus the metadata needed to cite it."""

    doc_name: str
    text: str
    page_start: int
    page_end: int  # equal to page_start, because chunks never span pages

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"Page {self.page_start}"
        return f"Pages {self.page_start}-{self.page_end}"


def build_chunks(
    pages: Iterable[PageText],
    doc_name: str,
    chunk_words: int = 120,
    overlap_words: int = 25,
) -> list[Chunk]:
    """Split each page into windows of chunk_words words. Windows on the same page share overlap_words words."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero.")
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be zero or more, and smaller than chunk_words.")

    step = chunk_words - overlap_words
    chunks: list[Chunk] = []
    for page in pages:
        words = page.text.split()
        start = 0
        while start < len(words):
            window = words[start:start + chunk_words]
            chunks.append(
                Chunk(
                    doc_name=doc_name,
                    text=" ".join(window),
                    page_start=page.page_number,
                    page_end=page.page_number,
                )
            )
            if start + chunk_words >= len(words):
                break
            start += step
    return chunks


def build_summary_source(pages: list[PageText], max_chars: int) -> tuple[str, list[int]]:
    """Join page texts with [Page N] markers, stopping before max_chars.

    Returns the joined text and the page numbers that were included.
    """
    parts: list[str] = []
    included: list[int] = []
    used = 0
    for page in pages:
        block = f"[Page {page.page_number}]\n{page.text}\n\n"
        if included and used + len(block) > max_chars:
            break
        parts.append(block)
        included.append(page.page_number)
        used += len(block)
    return "".join(parts), included
