"""
A minimal local vector store: fitted embeddings + clause metadata,
persisted to disk as a single .npz + a sidecar JSON for metadata.

This plays the architectural role the brief assigns to ChromaDB. It is
deliberately not ChromaDB (see DECISIONS.md) but it exposes the same shape
of operation -- build once, query many times, keep vectors and metadata
together -- so that swapping in a real vector database later means
reimplementing this one file, not touching retrieval/vector_search.py or
anything above it.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.ingestion.chunker import Clause
from app.retrieval.embeddings import EmbeddingBackend, get_default_backend


class VectorStore:
    def __init__(self, backend: EmbeddingBackend | None = None):
        self.backend = backend or get_default_backend()
        self.clauses: list[Clause] = []
        self.matrix: np.ndarray | None = None

    def build(self, clauses: list[Clause]) -> None:
        self.clauses = clauses
        texts = [self._embed_text(c) for c in clauses]
        self.backend.fit(texts)
        self.matrix = self.backend.transform(texts)

    @staticmethod
    def _embed_text(clause: Clause) -> str:
        # Fold in the section title -- it carries topical vocabulary
        # ("Recipient obligations", "Overpayments and Recovery") that the
        # clause body itself sometimes doesn't repeat, which helps recall
        # on paraphrased questions.
        return f"{clause.section_title}. {clause.text}"

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "matrix.npy", self.matrix)
        with open(path / "backend.pkl", "wb") as f:
            pickle.dump(self.backend, f)
        with open(path / "clauses.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.clauses], f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        if not (path / "matrix.npy").exists():
            raise FileNotFoundError(
                f"No index found at {path}. Run `python scripts/ingest.py` first."
            )
        store = cls()
        store.matrix = np.load(path / "matrix.npy")
        with open(path / "backend.pkl", "rb") as f:
            store.backend = pickle.load(f)
        with open(path / "clauses.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        store.clauses = [Clause(**c) for c in raw]
        return store
