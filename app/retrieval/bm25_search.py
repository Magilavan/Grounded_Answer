"""
BM25 keyword search.

Implemented directly (the standard Okapi BM25 formula) rather than via the
`rank-bm25` package, purely because this project targets a clean-clone
environment and BM25 is about 40 lines of unambiguous, well-specified
arithmetic -- pinning one more third-party dependency for it isn't worth
the added install surface. Swapping in `rank-bm25` later is a same-shape
change confined to this file.

BM25 is what catches exact clause references, dollar figures, day counts,
and specific policy terms that a TF-IDF/cosine score can under-weight --
see DECISIONS.md for why both retrievers are kept and fused rather than
picking one.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.ingestion.chunker import Clause
from app.retrieval.vector_search import SearchHit

TOKEN_RE = re.compile(r"[a-z0-9§]+(?:\.[a-z0-9]+)*")


def _stem_token(token: str) -> str:
    if len(token) <= 4:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("ations"):
        return token[:-1]
    if token.endswith("es") and token[-3] in "sxz":
        return token[:-2]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    raw = TOKEN_RE.findall(text.lower())
    return [_stem_token(t) for t in raw]


class BM25Index:
    """Okapi BM25 with the standard k1=1.5, b=0.75 defaults."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.clauses: list[Clause] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Counter = Counter()  # document frequency per term
        self.n_docs: int = 0

    def build(self, clauses: list[Clause]) -> None:
        self.clauses = clauses
        self.doc_tokens = [
            tokenize(f"{c.citation} {c.section_title} {c.text}") for c in clauses
        ]
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.n_docs = len(clauses)
        self.avg_doc_len = sum(self.doc_lens) / self.n_docs if self.n_docs else 0.0

        self.df = Counter()
        for freqs in self.doc_freqs:
            for term in freqs:
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # Standard BM25 IDF with a +1 smoothing to keep it non-negative for
        # terms that appear in most documents.
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[float]:
        q_tokens = tokenize(query)
        scores = [0.0] * self.n_docs
        for term in set(q_tokens):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i in range(self.n_docs):
                f = self.doc_freqs[i].get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_doc_len or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


def bm25_search(index: BM25Index, query: str, top_k: int) -> list[SearchHit]:
    scores = index.score(query)
    ranked = sorted(range(len(index.clauses)), key=lambda i: scores[i], reverse=True)[
        :top_k
    ]
    return [
        SearchHit(clause=index.clauses[i], score=scores[i], retriever="bm25")
        for i in ranked
        if scores[i] > 0
    ]
