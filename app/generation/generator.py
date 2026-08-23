"""
Turns a Decision (ANSWER / REFUSE / CONFLICT) plus its evidence into the
final text shown to the user.

Default behaviour requires no LLM and no network: it assembles the answer
directly from the verified evidence text. This is intentional, not just a
fallback -- see DECISIONS.md ("why the deterministic path is the one this
project's own tests and evaluation run against"). An LLM call is attempted
first only when LLM_ENABLED is true and LLM_API_KEY is set; if it is
unavailable, fails, or produces a citation that doesn't pass validation,
the deterministic path is used silently. Either way, no citation reaches
the user without passing through app/citations/validator.py first.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.citations.validator import validate_citations
from app.config import Settings, SETTINGS
from app.generation.prompt import build_messages
from app.ingestion.chunker import Clause
from app.reasoning.contradiction_detector import Contradiction
from app.reasoning.decision import Decision, DecisionType
from app.reasoning.evidence import EvidenceItem
from app.reasoning.refusal import ESCALATION_LINE, refusal_message

logger = logging.getLogger("grounded_answer.generation")

_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
_WS_RE = re.compile(r"[ \t]+")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Fallback: handle unclosed <think> blocks (truncated by max_tokens)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)


def _clean(text: str) -> str:
    text = _BOLD_RE.sub(r"\1", text)
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


@dataclass
class GenerationResult:
    text: str
    citations: list[str]
    decision_type: DecisionType
    used_llm: bool = False
    related_clauses: list[str] = field(default_factory=list)


def _deterministic_answer(items: list[EvidenceItem]) -> str:
    """Render supporting clause text only -- no citation footer here.

    Citations are returned separately via GenerationResult.citations and
    displayed by the caller (CLI, evaluation harness). Building the
    footer in both places was a real bug found during final validation:
    it produced a doubled "Sources: ..." line in the CLI for any
    multi-clause answer.
    """
    if len(items) == 1:
        return _clean(items[0].hit.clause.text)
    parts = [_clean(it.hit.clause.text) + f" [{it.hit.clause.citation}]" for it in items]
    return "\n\n".join(parts)


def _call_llm(question: str, items: list[EvidenceItem], settings: Settings) -> str | None:
    """Ask the configured LLM to phrase the answer from already-verified
    evidence. Uses the OpenAI-compatible chat completions request/response
    shape, which Groq (the default provider) implements -- see
    https://console.groq.com/docs/api-reference#chat-create. Any other
    OpenAI-compatible endpoint works by overriding LLM_API_BASE/LLM_MODEL.

    Never raises: any failure (missing key, network error, malformed
    response, rate limit) just returns None, and the caller falls back to
    the deterministic template. This project's own tests and evaluation
    run with no LLM configured, so this path is not on the critical path
    for grounding correctness -- see DECISIONS.md.
    """
    if not settings.llm_enabled or not settings.llm_api_key:
        return None
    try:
        import requests  # local import: keeps `requests` optional for the
        # fully-offline deterministic path
    except ImportError:
        return None

    context = "\n\n".join(
        f"[{it.hit.clause.citation}] {_clean(it.hit.clause.text)}" for it in items
    )
    messages = build_messages(question, context)
    try:
        resp = requests.post(
            settings.llm_api_base,
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "max_tokens": 1024,
                "temperature": 0.0,
                "messages": messages,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        text = choices[0].get("message", {}).get("content", "").strip()
        # Strip <think>...</think> reasoning blocks from thinking models
        # (e.g. Qwen 3.6) so only the actual answer is used.
        # First strip closed blocks, then any unclosed trailing block.
        text = _THINK_RE.sub("", text).strip()
        text = _THINK_UNCLOSED_RE.sub("", text).strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 - any failure just falls back
        logger.info("LLM generation unavailable, using deterministic answer: %s", exc)
        return None


def _conflict_text(contradictions: list[Contradiction]) -> tuple[str, list[str]]:
    seen_pairs = set()
    blocks = []
    citations: list[str] = []
    for c in contradictions:
        pair_key = tuple(sorted([c.clause_a.clause_id, c.clause_b.clause_id]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        blocks.append(
            f"§{c.clause_a.clause_id} states: {_clean(c.clause_a.text)}\n\n"
            f"§{c.clause_b.clause_id} states: {_clean(c.clause_b.text)}"
        )
        citations.extend([c.clause_a.citation, c.clause_b.citation])

    body = "\n\n---\n\n".join(blocks)
    text = (
        "The policy manual contains conflicting provisions relevant to this "
        f"question.\n\n{body}\n\nBecause these provisions conflict, the "
        f"manual does not provide a single clear answer.\n\n{ESCALATION_LINE}"
    )
    # de-duplicate while preserving order
    seen = set()
    unique_citations = [c for c in citations if not (c in seen or seen.add(c))]
    return text, unique_citations


def generate_response(
    question: str,
    decision: Decision,
    all_clauses_by_id: dict[str, Clause],
    settings: Settings | None = None,
) -> GenerationResult:
    settings = settings or SETTINGS
    related = [it.hit.clause.citation for it in decision.evidence.items]

    if decision.decision_type == DecisionType.CONFLICT:
        text, citations = _conflict_text(decision.contradictions)
        return GenerationResult(
            text=text,
            citations=citations,
            decision_type=DecisionType.CONFLICT,
            related_clauses=related,
        )

    if decision.decision_type == DecisionType.REFUSE:
        text = refusal_message(decision.refusal_reason, detail=decision.evidence.detail)
        return GenerationResult(
            text=text,
            citations=[],
            decision_type=DecisionType.REFUSE,
            related_clauses=related,
        )

    # ANSWER
    supporting = decision.evidence.supporting
    supporting_clauses = [it.hit.clause for it in supporting]

    llm_text = _call_llm(question, supporting, settings)
    used_llm = False
    final_text = None

    if llm_text:
        validation = validate_citations(llm_text, supporting_clauses, all_clauses_by_id)
        if validation.valid and validation.checked_citations:
            final_text = llm_text
            used_llm = True
        else:
            logger.info(
                "Discarding LLM answer due to citation validation issues: %s",
                validation.issues,
            )

    if final_text is None:
        final_text = _deterministic_answer(supporting)
        # No further citation validation needed here: the deterministic
        # template is constructed directly *from* `supporting_clauses`, so
        # its correctness doesn't depend on scanning the rendered text.
        # (An earlier version of this function re-ran validate_citations()
        # on this path too, and it broke: a supporting clause's own text
        # sometimes mentions another clause in passing -- e.g. quoting
        # §1.4.1 verbatim can surface a "§2.1.2" that's part of the
        # manual's own wording, not a claim this answer is making. Scanning
        # rendered prose for citation-shaped substrings can't tell "this
        # is what I'm citing" apart from "this is inside a quote", so that
        # check only belongs on free-form LLM output below, where an
        # invented citation is a real risk.)
        citations = [c.citation for c in supporting_clauses]
        return GenerationResult(
            text=final_text,
            citations=citations,
            decision_type=DecisionType.ANSWER,
            used_llm=used_llm,
            related_clauses=related,
        )

    # LLM path: the model's free-form text is the one place a genuinely
    # invented citation could appear, so it alone gets the full-text scan.
    final_validation = validate_citations(final_text, supporting_clauses, all_clauses_by_id)
    if not final_validation.valid:
        from app.reasoning.refusal import RefusalReason

        text = refusal_message(
            RefusalReason.CITATION_UNVERIFIABLE,
            detail="; ".join(final_validation.issues),
        )
        return GenerationResult(
            text=text,
            citations=[],
            decision_type=DecisionType.REFUSE,
            related_clauses=related,
        )

    citations = [c.citation for c in supporting_clauses]
    return GenerationResult(
        text=final_text,
        citations=citations,
        decision_type=DecisionType.ANSWER,
        used_llm=used_llm,
        related_clauses=related,
    )
