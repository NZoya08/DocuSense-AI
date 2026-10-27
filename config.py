"""Central settings for DocuSense AI.

Each value can be overridden with an environment variable. Locally, put them in a .env file
(copy .env.example to .env). On Streamlit Community Cloud, add them under the app's Secrets.
Never hard-code API keys in this file.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv reads a local .env file. It is optional.
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


def get_setting(name: str, default: str | None = None) -> str | None:
    """Return a setting from environment variables, then Streamlit secrets, then the default."""
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    try:
        import streamlit as st

        if name in st.secrets:
            secret_value = str(st.secrets[name]).strip()
            if secret_value:
                return secret_value
    except Exception:  # No secrets file, or Streamlit is not installed (for example in unit tests)
        pass
    return default


APP_NAME = "DocuSense AI"
APP_SUBTITLE = "Evidence-Grounded Document Intelligence Assistant"

# Upload limits
MAX_UPLOAD_MB = 10
MAX_PAGES = 200
MIN_TEXT_CHARS = 40  # less selectable text than this counts as "no usable text"

# Chunking, measured in words. all-MiniLM-L6-v2 embeds only about the first 256 word
# pieces of each input, so chunks stay well below that size.
CHUNK_WORDS = 120
CHUNK_OVERLAP_WORDS = 25

# Retrieval
TOP_K = 4
EMBEDDING_BACKEND = (get_setting("EMBEDDING_BACKEND", "semantic") or "semantic").lower()
EMBEDDING_MODEL = get_setting("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# A passage must reach this match score to count as evidence. These are starting points,
# not measured optimums. Tune them on your own documents.
MIN_RELEVANCE_SCORE = {"semantic": 0.30, "keyword": 0.05}

# Gemini
GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-2.5-flash")
TEMPERATURE = 0.2
ANSWER_MAX_OUTPUT_TOKENS = 900
SUMMARY_MAX_OUTPUT_TOKENS = 2048
MAX_QUESTION_CHARS = 500
MAX_SUMMARY_CHARS = 40000  # document text sent for a summary

if EMBEDDING_BACKEND not in ("semantic", "keyword"):
    raise ValueError("EMBEDDING_BACKEND must be 'semantic' or 'keyword'.")


def get_gemini_api_key() -> str | None:
    """Return the Gemini API key, or None when it has not been configured."""
    key = get_setting("GEMINI_API_KEY")
    if not key or key.startswith("paste-your"):  # the placeholder from .env.example
        return None
    return key
