"""
Reranking of fused candidates.

The brief recommends a cross-encoder here. We don't ship one (see
DECISIONS.md -- no network-fetched model weights in a clean-clone
project), but the *role* the reranker plays in the pipeline is preserved
exactly: it looks at (question, passage) pairs together, not each
independently, and produces a relevance score used only for ordering and
gating -- never as a stand-in for evidence verification (section 13/14 of
the brief). That separation is enforced structurally: this module has no
knowledge of ANSWER/REFUSE/CONFLICT, it only ranks.

The heuristic combines:
  - IDF-weighted, lightly-stemmed token overlap between question and
    passage
  - a bonus when a number mentioned in the question (a day count, a
    dollar figure) also appears in the passage
  - a bonus when the question explicitly names a clause (e.g. "§4.3.2")
    and the passage is that clause

Two corrections were needed after the first version, both found by
actually running evaluation questions rather than assumed up front (see
DECISIONS.md for the full account):

  1. Light stemming. "What is the definition of X" vs. a clause under a
     section titled "Definitions" don't share a token without it --
     unigram bag-of-words has no notion that "definition" and
     "Definitions" are the same idea. A full stemmer (Porter etc.) is
     more than this corpus needs; a handful of suffix rules gets the
     real cases (definition/definitions, resource/resources,
     condition/conditions) without the risk of a heavier stemmer
     collapsing unrelated words together.

  2. Corpus-native IDF that also excludes named-entity boilerplate.
     "Calder County" is the name of the jurisdiction the whole manual is
     about -- semantically it carries no topical signal for retrieval --
     but in this small, ~150-clause corpus it happens to appear in only
     ~5-8% of clauses (most clauses just say "the Department" or "a
     recipient" without restating the county name), which gives it a
     statistically high IDF. A query like "speed limit in Calder County"
     was scoring a completely unrelated eligibility clause as highly
     relevant purely because both mention "Calder County". The fix is a
     short, explicitly-justified list of jurisdiction/manual-identity
     terms held out of the weighting, the same way "the"/"a" are held out
     for being grammatically rather than topically frequent.

`CrossEncoderReranker` is the seam for swapping in a real cross-encoder
(e.g. sentence-transformers CrossEncoder) later without touching anything
that calls `rerank()`.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter

from app.ingestion.chunker import Clause
from app.retrieval.bm25_search import tokenize
from app.retrieval.vector_search import SearchHit

CLAUSE_REF_RE = re.compile(r"§?\s*(\d+\.\d+\.\d+)")
NUMBER_RE = re.compile(r"\b\d[\d,]*\b")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "and", "or", "if", "do", "does", "did",
    "what", "when", "how", "can", "could", "would", "should", "i", "my",
    "me", "it", "this", "that", "with", "as", "at", "by", "from",
    "have", "has", "had", "having", "they", "them", "their", "there",
    "many", "much", "long", "more", "most", "who", "whom", "whose", "which",
}

# Jurisdiction/manual-identity terms: statistically sparse in this small
# corpus (so IDF would rank them as "distinctive") but semantically inert
# for topic matching -- see module docstring, point 2. Kept short and
# explicit rather than a generic named-entity detector, since this is a
# single fixed manual, not a changing corpus.
DOMAIN_BOILERPLATE = {"calder", "county", "manual"}

ALL_STOPWORDS = STOPWORDS | DOMAIN_BOILERPLATE


def _stem(token: str) -> str:
    """A handful of suffix rules, not a real stemmer.

    Deliberately conservative: only strips endings that are unambiguous
    in this policy-manual vocabulary, to avoid collapsing distinct words
    together (a real Porter/Snowball stemmer is available if this ever
    needs to generalise beyond this manual).
    """
    if len(token) <= 4:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("ations"):
        return token[:-1]  # "determinations" -> "determination"
    if token.endswith("es") and token[-3] in "sxz":
        return token[:-2]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def stemmed_tokens(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text) if t not in ALL_STOPWORDS]


class Reranker(ABC):
    @abstractmethod
    def rerank(self, question: str, hits: list[SearchHit]) -> list[tuple[SearchHit, float]]:
        """Return (hit, relevance_score) pairs, score in [0, 1], sorted desc."""


class HeuristicReranker(Reranker):
    """See module docstring for the overall approach and the two fixes
    (stemming, boilerplate exclusion) folded in after evaluation runs
    surfaced them."""

    def __init__(self, idf: dict[str, float] | None = None):
        # idf: stemmed token -> inverse document frequency, computed from
        # the corpus at index build time (see build_idf_lookup below). A
        # token missing from the corpus vocabulary (e.g. a typo, or a word
        # from a wildly out-of-scope question) gets a high default weight
        # so its ABSENCE from a passage isn't silently ignored, but its
        # presence in a passage still requires an actual corpus match to
        # score at all.
        self.idf = idf or {}
        self._default_idf = max(self.idf.values(), default=1.0)

    def _weight(self, token: str) -> float:
        return self.idf.get(token, self._default_idf)

    def rerank(self, question: str, hits: list[SearchHit]) -> list[tuple[SearchHit, float]]:
        q_token_set = set(stemmed_tokens(question))
        q_numbers = set(NUMBER_RE.findall(question))
        q_clause_refs = set(CLAUSE_REF_RE.findall(question))

        q_weight_total = sum(self._weight(t) for t in q_token_set) or 1.0

        scored: list[tuple[SearchHit, float]] = []
        for hit in hits:
            passage = f"{hit.clause.section_title} {hit.clause.text}"
            p_token_set = set(stemmed_tokens(passage))

            matched = q_token_set & p_token_set
            matched_weight = sum(self._weight(t) for t in matched)
            overlap = matched_weight / q_weight_total

            # A single incidental word match is noise, not evidence of
            # relevance -- require at least two shared content words
            # before overlap counts, unless the question itself only has
            # one content word to give.
            if len(matched) < 2 and len(q_token_set) > 1:
                overlap = 0.0

            number_bonus = 0.0
            if q_numbers:
                p_numbers = set(NUMBER_RE.findall(passage))
                if q_numbers & p_numbers:
                    number_bonus = 0.2

            clause_bonus = 0.0
            if q_clause_refs and hit.clause.clause_id in q_clause_refs:
                clause_bonus = 0.4

            raw = 0.7 * overlap + number_bonus + clause_bonus
            scored.append((hit, min(raw, 1.0)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored


def get_default_reranker(idf: dict[str, float] | None = None) -> Reranker:
    return HeuristicReranker(idf=idf)


def build_idf_lookup(clauses: list[Clause]) -> dict[str, float]:
    """Compute a stemmed-token -> IDF mapping directly from the clause
    corpus, using the same stemming and boilerplate exclusion the
    reranker itself applies at query time.

    Computed natively here (rather than reused from the vector store's
    TfidfVectorizer) so the vocabulary the reranker weights against is
    guaranteed to be built with identical stemming/stopword rules -- the
    two were drifting apart when the reranker used the vectorizer's raw,
    unstemmed vocabulary.
    """
    n_docs = len(clauses)
    df: Counter = Counter()
    for c in clauses:
        toks = set(stemmed_tokens(f"{c.section_title} {c.text}"))
        for t in toks:
            df[t] += 1

    return {
        term: math.log((n_docs + 1) / (freq + 1)) + 1.0 for term, freq in df.items()
    }
