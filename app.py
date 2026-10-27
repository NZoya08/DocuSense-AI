"""DocuSense AI: the Streamlit user interface.

Run it from the project folder with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import traceback

import streamlit as st

from chunker import build_chunks
from config import (
    APP_NAME,
    APP_SUBTITLE,
    CHUNK_OVERLAP_WORDS,
    CHUNK_WORDS,
    EMBEDDING_BACKEND,
    GEMINI_MODEL,
    MAX_PAGES,
    MAX_UPLOAD_MB,
    get_gemini_api_key,
)
from llm_service import GeminiService, LLMServiceError
from pdf_processor import DocumentError, extract_document
from rag_pipeline import (
    answer_question,
    build_summary_file,
    format_page_list,
    safe_file_stem,
    summarize_document,
)
from retriever import DocumentIndex, create_embedder

st.set_page_config(page_title=APP_NAME, layout="wide")

THEME_CSS = """
<style>
.block-container { max-width: 1140px; padding-top: 2rem; }
.ds-hero { background: #0B1F3A; color: #FFFFFF; border-radius: 12px; padding: 1.5rem 1.75rem; margin-bottom: 1.25rem; }
.ds-hero .ds-subtitle { color: #B8C7DB; font-size: 0.95rem; margin-bottom: 0.2rem; }
.ds-hero h1 { color: #FFFFFF; font-size: 2rem; margin: 0 0 0.4rem 0; }
.ds-hero p { color: #DCE5F2; margin: 0; line-height: 1.55; }
.ds-empty { background: #F3F5F9; border: 1px dashed #B9C3D3; border-radius: 10px; padding: 0.9rem 1.1rem; color: #33445E; }
.ds-passage-label { color: #1F4E79; font-weight: 600; font-size: 0.9rem; margin-top: 0.7rem; }
.ds-passage { background: #F3F5F9; border-left: 3px solid #1F4E79; border-radius: 6px; padding: 0.7rem 0.9rem; color: #0B1F3A; line-height: 1.55; font-size: 0.93rem; }
div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #D9DEE7; border-radius: 10px; padding: 0.75rem 1rem; }
section[data-testid="stSidebar"] { background: #F3F5F9; }
footer { visibility: hidden; }
</style>
"""


def display_text(text: str) -> str:
    """Escape dollar signs so Streamlit does not show the text between two of them as a maths formula."""
    return text.replace("$", r"\$")


def init_state() -> None:
    defaults = {
        "document": None,  # ExtractedDocument for the current upload
        "index": None,  # DocumentIndex for the current upload
        "chunk_count": 0,
        "retrieval_label": "",
        "retrieval_notice": None,
        "upload_error": None,
        "document_hash": None,
        "summary": None,
        "messages": [],
        "uploader_version": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_document() -> None:
    """Reset everything tied to the current document. Runs as a button callback."""
    st.session_state.document = None
    st.session_state.index = None
    st.session_state.chunk_count = 0
    st.session_state.retrieval_label = ""
    st.session_state.retrieval_notice = None
    st.session_state.upload_error = None
    st.session_state.document_hash = None
    st.session_state.summary = None
    st.session_state.messages = []
    st.session_state.uploader_version += 1  # a new widget key gives the file uploader an empty state


def _store_failure(message: str) -> None:
    st.session_state.document = None
    st.session_state.index = None
    st.session_state.chunk_count = 0
    st.session_state.upload_error = message


def process_upload(file_name: str, data: bytes, digest: str) -> None:
    """Extract text, split it into chunks and build the search index. Errors are stored for display."""
    # Remember the file even if it fails, so the same bad file is not reprocessed on every rerun.
    st.session_state.document_hash = digest
    st.session_state.upload_error = None
    st.session_state.summary = None
    st.session_state.messages = []
    try:
        with st.spinner("Checking the PDF and extracting text..."):
            document = extract_document(file_name, data)
            chunks = build_chunks(
                document.pages,
                document.name,
                chunk_words=CHUNK_WORDS,
                overlap_words=CHUNK_OVERLAP_WORDS,
            )
        with st.spinner("Building the search index. The first run can take a minute while the model loads..."):
            embedder, notice = create_embedder(EMBEDDING_BACKEND)
            index = DocumentIndex(embedder)
            index.build(chunks)
    except DocumentError as exc:
        _store_failure(str(exc))
        return
    except Exception:  # keep the app running, and show a calm message
        traceback.print_exc()
        _store_failure("Something went wrong while processing this PDF. Try another file.")
        return

    st.session_state.document = document
    st.session_state.index = index
    st.session_state.chunk_count = len(chunks)
    st.session_state.retrieval_label = embedder.label
    st.session_state.retrieval_notice = notice


def build_service(api_key: str) -> GeminiService | None:
    try:
        return GeminiService(api_key=api_key, model_name=GEMINI_MODEL)
    except Exception as exc:  # for example, the google-genai package is not installed
        st.error(f"The Gemini client could not start: {exc}")
        return None


def run_question(question: str, index, service) -> dict:
    """Answer one question and return a chat message dictionary ready for display."""
    try:
        result = answer_question(index, service, question)
    except ValueError as exc:
        return {"role": "assistant", "kind": "warning", "content": str(exc)}
    except LLMServiceError as exc:
        return {"role": "assistant", "kind": "error", "content": str(exc)}
    except Exception:
        traceback.print_exc()
        return {"role": "assistant", "kind": "error", "content": "Something went wrong while answering. Please try again."}

    if not result.answered:
        return {"role": "assistant", "kind": "info", "content": result.message}
    return {
        "role": "assistant",
        "kind": "answer",
        "content": result.answer_text,
        "cited": result.cited,
        "sources": result.sources,
    }


def render_hero() -> None:
    st.markdown(
        '<div class="ds-hero">'
        f'<div class="ds-subtitle">{html.escape(APP_SUBTITLE)}</div>'
        f"<h1>{html.escape(APP_NAME)}</h1>"
        "<p>Upload a text-based PDF, ask questions in plain English, and check every answer against the "
        "page-numbered passages it was based on.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    document = st.session_state.document
    with st.sidebar:
        st.markdown("#### Document details")
        if document is None:
            st.caption("No document loaded yet.")
        else:
            st.markdown(f"**File:** {html.escape(document.name)}")
            st.markdown(f"**Pages:** {document.page_count}, with text: {document.text_page_count}")
            st.markdown(f"**Text chunks:** {st.session_state.chunk_count}")
            st.markdown(f"**Search:** {html.escape(st.session_state.retrieval_label)}")
        st.markdown(f"**Model:** {html.escape(GEMINI_MODEL)}")

        st.markdown("#### How to use")
        st.markdown(
            f"Upload a text-based PDF of up to {MAX_UPLOAD_MB} MB and {MAX_PAGES} pages. "
            "Check the status, then generate a summary or ask a question. "
            "Open the sources to read the passages that each answer used."
        )

        st.markdown("#### Privacy")
        st.caption(
            "Questions and summaries send relevant passages of your document to Google's Gemini API. "
            "Do not upload confidential documents unless you understand the provider's data terms."
        )
        st.button("Clear document and chat", on_click=clear_document)


def render_upload_section() -> None:
    st.markdown("#### Upload")
    st.caption(
        f"Text-based PDFs only, up to {MAX_UPLOAD_MB} MB and {MAX_PAGES} pages. Pages without selectable "
        "text cannot be read, because OCR is not supported in this version."
    )
    uploaded = st.file_uploader("PDF file", type=["pdf"], key=f"uploader_{st.session_state.uploader_version}")
    if uploaded is not None:
        data = uploaded.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        if digest != st.session_state.document_hash:
            process_upload(uploaded.name, data, digest)


def render_status_section() -> None:
    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)
    document = st.session_state.document
    if document is None:
        if not st.session_state.upload_error:
            st.markdown(
                '<div class="ds-empty">Upload a PDF to see its page count, extraction status and search method.</div>',
                unsafe_allow_html=True,
            )
        return

    st.markdown("#### Document status")
    columns = st.columns(4)
    columns[0].metric("Pages", document.page_count)
    columns[1].metric("Pages with text", document.text_page_count)
    columns[2].metric("Text chunks", st.session_state.chunk_count)
    method = "Semantic" if st.session_state.index.embedder.kind == "semantic" else "Keyword"
    columns[3].metric("Search method", method)

    if document.is_partial:
        st.warning(
            f"{document.status_text}. Pages {format_page_list(document.skipped_pages)} cannot be used "
            "for answers or summaries."
        )
    else:
        st.success(f"{document.status_text}.")
    if st.session_state.retrieval_notice:
        st.info(st.session_state.retrieval_notice)


def render_summary_section(service) -> None:
    st.markdown("#### Summary")
    document = st.session_state.document
    if document is None:
        st.caption("Upload a document to generate a structured summary.")
        return
    if service is None:
        st.caption("Add a GEMINI_API_KEY to generate summaries.")
        return

    label = "Regenerate summary" if st.session_state.summary else "Generate summary"
    if st.button(label, type="primary"):
        try:
            with st.spinner("Writing the summary..."):
                st.session_state.summary = summarize_document(service, document)
        except LLMServiceError as exc:
            st.error(str(exc))

    summary = st.session_state.summary
    if summary is None:
        st.caption("The summary covers the main purpose, key points, important dates and requirements, and a conclusion.")
        return
    for note in summary.incomplete_notes:
        st.warning(note)
    st.markdown(display_text(summary.markdown))
    st.download_button(
        "Download summary (.md)",
        data=build_summary_file(document.name, summary),
        file_name=f"{safe_file_stem(document.name)}_summary.md",
        mime="text/markdown",
    )


def render_sources(sources) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(
                f'<div class="ds-passage-label">[{source.number}] {html.escape(source.page_label)}, '
                f"match score {source.score:.2f}</div>"
                f'<div class="ds-passage">{html.escape(source.text).replace("$", "&#36;")}</div>',
                unsafe_allow_html=True,
            )


def render_message_content(message: dict) -> None:
    kind = message.get("kind", "text")
    if message["role"] == "user" or kind == "text":
        st.markdown(display_text(message["content"]))
    elif kind == "answer":
        st.markdown(display_text(message["content"]))
        if not message.get("cited"):
            st.caption("The model did not cite passage numbers in this answer. Check the sources before relying on it.")
        render_sources(message["sources"])
    elif kind == "error":
        st.error(message["content"])
    elif kind == "info":
        st.info(message["content"])
    else:
        st.warning(message["content"])


def render_chat_section(service) -> None:
    st.markdown("#### Ask a question")
    document = st.session_state.document
    ready = document is not None and service is not None
    if document is None:
        placeholder = "Upload a PDF to start asking questions"
    elif service is None:
        placeholder = "Add a GEMINI_API_KEY to enable answers"
    else:
        placeholder = "Ask a question about this document"

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_message_content(message)

    question = st.chat_input(placeholder, disabled=not ready)
    if question and ready:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(display_text(question))
        with st.chat_message("assistant"):
            with st.spinner("Searching the document and writing an answer..."):
                reply = run_question(question, st.session_state.index, service)
            render_message_content(reply)
        st.session_state.messages.append(reply)


def main() -> None:
    init_state()
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    render_hero()
    render_sidebar()

    api_key = get_gemini_api_key()
    service = None
    if api_key:
        service = build_service(api_key)
    else:
        st.warning(
            "GEMINI_API_KEY is not set. You can still upload and check PDFs, but answers and summaries "
            "are turned off until a key is added. See the README for setup."
        )

    render_upload_section()
    render_status_section()
    render_summary_section(service)
    render_chat_section(service)


main()
