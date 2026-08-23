#!/usr/bin/env python3
"""
Interactive CLI for the Grounded Answer assistant.

    python -m app.cli

Type 'exit' or 'quit' to leave, or 'verbose' to toggle showing retrieval/
decision detail alongside each answer.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SETTINGS
from app.ingestion.loader import ManualNotFoundError
from app.pipeline import EmptyQuestionError, PipelineContext, answer_question
from app.reasoning.decision import DecisionType

BANNER = """==================================================
 Brite Spark 2026
 The Grounded Answer
==================================================

Ask a policy question.
Type 'exit' to quit, 'verbose' to toggle detail view.
"""


def _print_verbose(result) -> None:
    print()
    if result.date_context and result.date_context.has_any_date:
        print(f"  [date context] primary={result.date_context.primary_date} "
              f"explicit={result.date_context.date_type_explicit} "
              f"spans_boundary={result.date_context.spans_boundary}")
    if result.temporal_resolution and result.temporal_resolution.notes:
        for note in result.temporal_resolution.notes:
            print(f"  [temporal note] {note}")
    print(f"  [retrieval] vector={result.vector_hit_count} "
          f"bm25={result.bm25_hit_count} fused={result.fused_hit_count}")
    print(f"  [decision]  {result.decision.decision_type.value}"
          + (f" ({result.decision.refusal_reason.value})"
             if result.decision.refusal_reason else ""))
    if result.decision.evidence.items:
        print("  [evidence considered]")
        for it in result.decision.evidence.items:
            flag = "OK" if it.supports_answer else (
                "DEFERRED-UNRESOLVED" if it.deferred_unresolved else "insufficient"
            )
            print(f"    {it.hit.clause.citation:>10}  relevance={it.relevance_score:.3f}  "
                  f"support={it.support_score:.3f}  [{flag}]")
    print()


def main() -> int:
    logging.basicConfig(
        level=getattr(logging, SETTINGS.log_level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stderr,
    )
    # Keep the CLI's own stdout clean; retrieval/decision logs go to
    # stderr so `python -m app.cli < questions.txt > answers.txt` stays
    # usable, per the brief's "do not expose technical detail to normal
    # users" guidance -- verbose mode surfaces the same detail on stdout
    # deliberately, on request.

    print(BANNER)
    try:
        ctx = PipelineContext.load()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Run `python scripts/ingest.py` first.", file=sys.stderr)
        return 1
    except ManualNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    verbose = False

    while True:
        try:
            question = input("Question:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            return 0
        if question.lower() == "verbose":
            verbose = not verbose
            print(f"(verbose mode {'on' if verbose else 'off'})\n")
            continue

        try:
            result = answer_question(question, ctx)
        except EmptyQuestionError:
            continue
        except Exception as exc:  # noqa: BLE001
            # Never show a raw traceback to a CLI user -- log it and give
            # a safe, generic message instead (brief section 35).
            logging.getLogger("grounded_answer.cli").exception(
                "Unhandled error answering question"
            )
            print("\nSomething went wrong answering that question. "
                  "Please try rephrasing it, or consult a policy "
                  "administrator directly.\n")
            continue

        print()
        print(f"Decision: {result.decision.decision_type.value}")
        print()
        print("Answer:")
        print(result.generation.text)
        print()
        if result.generation.citations:
            label = "Sources" if len(result.generation.citations) > 1 else "Source"
            print(f"{label}: {', '.join(result.generation.citations)}")
            print()

        if verbose:
            _print_verbose(result)


if __name__ == "__main__":
    raise SystemExit(main())
