"""Helpers that build small PDFs in memory for the tests. They use PyMuPDF, like the app does."""

import fitz  # PyMuPDF


def make_pdf(page_texts):
    """Build a PDF with one page per entry. A None entry creates a page with a drawing but no text."""
    document = fitz.open()
    try:
        for text in page_texts:
            page = document.new_page()
            if text is None:
                page.draw_rect(fitz.Rect(72, 72, 200, 200), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
            else:
                page.insert_textbox(fitz.Rect(72, 72, 523, 770), text, fontsize=11)
        return document.tobytes()
    finally:
        document.close()
