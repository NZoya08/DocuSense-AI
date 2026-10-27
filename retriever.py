"""Turns text into vectors and finds the passages closest to a question.

Two embedders share one interface (fit, encode, kind, label):

* SemanticEmbedder uses a pretrained Sentence-Transformers model. It compares meaning, so
  "holiday allowance" can match "annual leave". This is the intended retrieval method.
* KeywordEmbedder is a TF-IDF style fallback. It only matches shared words. The app uses it
  when the semantic model cannot be loaded, and it says so in the interface.

Vectors are scaled to unit length, so a dot product equals cosine similarity.
"""

from __future__ import annotations

import functools
import importlib.util
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from chunker import Chunk
from config import EMBEDDING_MODEL, TOP_K

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to "
    "was were what when where which who will with you your".split()
)
_MAX_VOCABULARY = 8000


def _tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN_PATTERN.findall(text.lower()):
        if len(token) < 2 or token in _STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]  # light plural handling, so "days" and "day" match
        tokens.append(token)
    return tokens


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class KeywordEmbedder:
    """TF-IDF style word vectors. A fallback: it matches shared words, not meaning."""

    kind = "keyword"
    label = "Keyword matching (fallback)"

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}
        self._idf = np.zeros(0, dtype=np.float32)

    def fit(self, texts: list[str]) -> None:
        document_frequency: Counter = Counter()
        for text in texts:
            document_frequency.update(set(_tokenize(text)))
        terms = sorted(term for term, _ in document_frequency.most_common(_MAX_VOCABULARY))
        self._vocabulary = {term: column for column, term in enumerate(terms)}
        n_documents = max(len(texts), 1)
        self._idf = np.array(
            [math.log((1 + n_documents) / (1 + document_frequency[term])) + 1.0 for term in terms],
            dtype=np.float32,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self._vocabulary)), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in Counter(_tokenize(text)).items():
                column = self._vocabulary.get(term)
                if column is not None:
                    matrix[row, column] = (1.0 + math.log(count)) * self._idf[column]
        return _normalize_rows(matrix)


@functools.lru_cache(maxsize=2)
def _load_sentence_model(model_name: str):
    # Imported here so the rest of the app still runs when the package is missing.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SemanticEmbedder:
    """Pretrained Sentence-Transformers embeddings. The model is downloaded once and cached."""

    kind = "semantic"

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.label = f"Semantic search ({model_name})"
        self._model = _load_sentence_model(model_name)

    def fit(self, texts: list[str]) -> None:
        return None  # A pretrained model needs nothing learned from the document.

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def create_embedder(backend: str):
    """Return (embedder, notice). The notice is set only when we fell back to keyword matching."""
    if backend == "keyword":
        return KeywordEmbedder(), None
    if backend != "semantic":
        raise ValueError("backend must be 'semantic' or 'keyword'.")
    if importlib.util.find_spec("sentence_transformers") is None:
        return KeywordEmbedder(), (
            "The sentence-transformers package is not installed, so keyword matching is used. "
            "Results are less accurate."
        )
    try:
        return SemanticEmbedder(), None
    except Exception as exc:  # download blocked, model file damaged, out of memory, and so on
        return KeywordEmbedder(), (
            f"The semantic model could not be loaded ({type(exc).__name__}), so keyword matching is used. "
            "Results are less accurate."
        )


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class DocumentIndex:
    """Holds one document's chunks and their vectors, and answers similarity searches."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index without text chunks.")
        texts = [chunk.text for chunk in chunks]
        self.embedder.fit(texts)
        self._matrix = self.embedder.encode(texts)
        self._chunks = list(chunks)

    def search(self, query: str, top_k: int = TOP_K, min_score: float = 0.0) -> list[RetrievedChunk]:
        cleaned = (query or "").strip()
        if not cleaned:
            raise ValueError("The question is empty.")
        if self._matrix is None:
            raise RuntimeError("Build the index before searching it.")
        query_vector = self.embedder.encode([cleaned])[0]
        scores = self._matrix @ query_vector  # cosine similarity, because all vectors are unit length
        results: list[RetrievedChunk] = []
        for position in np.argsort(-scores)[:top_k]:
            score = float(scores[position])
            if score <= 0.0 or score < min_score:
                continue
            results.append(RetrievedChunk(chunk=self._chunks[int(position)], score=score))
        return results
