#!/usr/bin/env python3
"""
Build the retrieval index from data/policy-manual.md.

Run this once before using the CLI or the evaluation harness, and again
any time the manual changes.

    python scripts/ingest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SETTINGS
from app.ingestion.chunker import parse_clauses
from app.ingestion.loader import ManualNotFoundError, load_manual_text
from app.retrieval.vector_store import VectorStore


def main() -> int:
    print("Loading policy manual...")
    try:
        text = load_manual_text(SETTINGS.policy_manual_path)
    except ManualNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Parsing clauses...")
    clauses = parse_clauses(text)
    if not clauses:
        print("ERROR: No clauses were parsed from the manual. Check its formatting.", file=sys.stderr)
        return 1
    print(f"  {len(clauses)} original clauses found across "
          f"{len({c.part_number for c in clauses})} parts.")

    print("Generating amendment clauses...")
    from app.ingestion.amendment_parser import generate_amendment_clauses
    amended_clauses = generate_amendment_clauses(clauses)
    all_clauses = clauses + amended_clauses
    print(f"  {len(amended_clauses)} amendment/transitional clauses generated. Total clauses: {len(all_clauses)}.")

    print("Building vector index...")
    store = VectorStore()
    store.build(all_clauses)

    print(f"Saving index to {SETTINGS.index_path} ...")
    store.save(SETTINGS.index_path)

    print("Done. Run the CLI with: python -m app.cli")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
