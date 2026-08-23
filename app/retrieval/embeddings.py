"""
Vectorisation for semantic-ish retrieval.

See DECISIONS.md ("Why TF-IDF instead of sentence-transformers") for the
full reasoning. In short: this corpus is ~150 short clauses, and a
dependency that requires downloading a neural embedding model at install
time is a real liability for a "clone and run from the README" deliverable
-- so the default backend is scikit-learn's TF-IDF + cosine similarity,
which needs nothing beyond what's already in requirements.txt and is
completely deterministic.

The module is written as a small interface (`EmbeddingBackend`) precisely
so that a real sentence-transformers backend can be dropped in later
(e.g. if the day-two change calls for it) without touching any other layer.
Anything importing this module should go through `get_default_backend()`
rather than importing TfidfEmbeddingBackend directly, so that swap stays
a one-line change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingBackend(ABC):
    """Interface every embedding backend must satisfy."""

    @abstractmethod
    def fit(self, documents: list[str]) -> None:
        """Fit the backend on the corpus (called once, at ingestion time)."""

    @abstractmethod
    def transform(self, documents: list[str]) -> np.ndarray:
        """Return a dense matrix of vectors, one row per document."""

    @abstractmethod
    def transform_query(self, query: str) -> np.ndarray:
        """Return a single vector for a query string."""


class TfidfEmbeddingBackend(EmbeddingBackend):
    """TF-IDF vectors as a stand-in for neural sentence embeddings.

    Word-level unigrams and bigrams, so that two-word policy terms
    ("countable income", "training allowance", "dependent child") get their
    own signal instead of being diluted across two unigram dimensions.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words="english",
        )
        self._fitted = False

    def fit(self, documents: list[str]) -> None:
        self.vectorizer.fit(documents)
        self._fitted = True

    def transform(self, documents: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EmbeddingBackend.fit() must be called before transform().")
        return self.vectorizer.transform(documents).toarray()

    def transform_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EmbeddingBackend.fit() must be called before transform_query().")
        return self.vectorizer.transform([query]).toarray()[0]


def get_default_backend() -> EmbeddingBackend:
    return TfidfEmbeddingBackend()


def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and a matrix of document
    vectors. Returns a 1-D array of scores, one per document. Handles the
    zero-vector case (e.g. a query with no vocabulary overlap) without
    raising a division-by-zero warning.
    """
    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(doc_matrix.shape[0])
    denom = doc_norms * query_norm
    denom[denom == 0] = 1e-12
    return (doc_matrix @ query_vec) / denom
