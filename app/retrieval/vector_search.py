"""Semantic (vector) search over the clause vector store."""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.chunker import Clause
from app.retrieval.embeddings import cosine_similarity
from app.retrieval.vector_store import VectorStore


@dataclass
class SearchHit:
    clause: Clause
    score: float
    retriever: str  # "vector" | "bm25"


def vector_search(store: VectorStore, query: str, top_k: int) -> list[SearchHit]:
    query_vec = store.backend.transform_query(query)
    scores = cosine_similarity(query_vec, store.matrix)
    ranked = sorted(
        range(len(store.clauses)), key=lambda i: scores[i], reverse=True
    )[:top_k]
    return [
        SearchHit(clause=store.clauses[i], score=float(scores[i]), retriever="vector")
        for i in ranked
        if scores[i] > 0
    ]
