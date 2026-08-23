"""
Evidence verification: does the retrieved, reranked evidence actually
*establish* an answer to the question, as opposed to merely being
topically related to it?

This is the layer the brief calls out as the most important one (section
14), and it is the layer that has to catch the "apparent gap" failure
mode: a passage that a naive retriever (and a naive human skimming it)
would treat as answering the question, but which -- read closely -- defers
the actual rule to somewhere else that doesn't in fact contain it.

The mechanism used here is generic, not keyed to this specific manual: any
clause whose text hands off the substantive answer to another clause (via
"see §X.X", "under §X.X", or "addressed separately") is checked for
whether that hand-off actually resolves. If the target clause doesn't
independently corroborate the question, or no target is even named, the
clause is marked as *relevant but not supporting* rather than treated as
sufficient evidence. This is exactly what should fire on §7.1.3 ("except
in the case of full-time students (see §5.4)" -- §5.4 is about care
allowances) and on §3.2.3 / §5.2.3 ("addressed separately" -- with no
provision anywhere that addresses it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingestion.chunker import Clause
from app.reasoning.refusal import RefusalReason
from app.retrieval.bm25_search import tokenize
from app.retrieval.vector_search import SearchHit

CLAUSE_REF_RE = re.compile(r"§\s*(\d+\.\d+(?:\.\d+)?)")

DEFERRAL_PHRASES = [
    "see §",
    "addressed separately",
    "dealt with separately",
    "treated separately",
    "provided for separately",
]

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "and", "or", "if", "do", "does", "did",
    "what", "when", "how", "can", "could", "would", "should", "i", "my",
    "me", "it", "this", "that", "with", "as", "at", "by", "from", "not",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and not t.isdigit()}


@dataclass
class EvidenceItem:
    hit: SearchHit
    relevance_score: float
    support_score: float
    supports_answer: bool
    deferred_unresolved: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class EvidenceAssessment:
    sufficient: bool
    items: list[EvidenceItem]  # all items considered, in score order
    supporting: list[EvidenceItem]  # subset judged to establish the answer
    reason: RefusalReason | None
    detail: str | None = None


def _find_deferral_targets(text: str) -> list[str]:
    """Return clause ids referenced via a deferral phrase in `text`.

    Only reports a clause id if it appears within ~40 characters of a
    deferral phrase, so that an ordinary informative cross-reference
    ("recovery under Part 9") isn't treated as a hand-off.
    """
    targets: list[str] = []
    lowered = text.lower()
    for phrase in DEFERRAL_PHRASES:
        idx = lowered.find(phrase)
        while idx != -1:
            window = text[idx : idx + 40]
            for m in CLAUSE_REF_RE.finditer(window):
                targets.append(m.group(1))
            idx = lowered.find(phrase, idx + 1)
    return targets


def _check_deferral(
    question: str, clause: Clause, clauses_by_id: dict[str, Clause]
) -> tuple[bool, list[str]]:
    """Check whether this clause defers the question's substance elsewhere,
    and if so, whether that hand-off actually resolves.

    Returns (unresolved, notes).
    """
    text = clause.text
    lowered = text.lower()

    has_bare_deferral = any(
        phrase in lowered and "§" not in text[max(0, lowered.find(phrase) - 5) : lowered.find(phrase) + 40]
        for phrase in ("addressed separately", "dealt with separately", "treated separately")
    )
    targets = _find_deferral_targets(text)

    if not targets and not has_bare_deferral:
        return False, []

    notes: list[str] = []
    q_tokens = _content_tokens(question)

    if not targets:
        # A deferral phrase with no named target at all -- the strongest
        # version of the "apparent gap" pattern: the manual promises
        # coverage "elsewhere" without saying where.
        notes.append(
            f"§{clause.clause_id} states the matter is addressed separately, "
            "but names no clause and no such provision was found elsewhere "
            "in the manual."
        )
        return True, notes

    # Deferral names a target: check whether the target clause actually
    # corroborates the question, rather than just existing.
    for target_id in targets:
        target = clauses_by_id.get(target_id) or _find_section_clauses(
            target_id, clauses_by_id
        )
        if not target:
            notes.append(
                f"§{clause.clause_id} refers to §{target_id}, which does not "
                "exist in the manual."
            )
            continue
        target_clauses = target if isinstance(target, list) else [target]
        target_tokens: set[str] = set()
        for tc in target_clauses:
            target_tokens |= _content_tokens(tc.text) | _content_tokens(tc.section_title)

        if not q_tokens:
            overlap = 0.0
        else:
            overlap = len(q_tokens & target_tokens) / len(q_tokens)

        if overlap < 0.15:
            preceding_text = _extract_preceding_condition(text, target_id)
            cond_tokens = _content_tokens(preceding_text) - {
                "see", "refer", "under", "clause", "section", "case", "except", "in", "the", "of", "to", "for"
            }
            if cond_tokens and not (q_tokens & cond_tokens):
                continue

            notes.append(
                f"§{clause.clause_id} defers to §{target_id} for this point, "
                f"but §{target_id} ({target_clauses[0].section_title}) does not "
                "actually address it."
            )
            return True, notes

    return False, notes


def _extract_preceding_condition(text: str, target_id: str) -> str:
    pos = text.find(target_id)
    if pos == -1:
        return ""
    start = max(0, pos - 80)
    snippet = text[start:pos]
    snippet = re.sub(r"\(?\s*(?:see|refer\s+to|under)\s*§?\s*$", "", snippet, flags=re.I)
    for sep in (".", ";", ","):
        if sep in snippet:
            snippet = snippet.split(sep)[-1]
    return snippet


def _find_section_clauses(
    ref: str, clauses_by_id: dict[str, Clause]
) -> list[Clause] | None:
    """`ref` may be a full clause id ("5.4.1") or a section id ("5.4").
    If it's a section id, return every clause in that section.
    """
    if ref in clauses_by_id:
        return [clauses_by_id[ref]]
    matches = [c for cid, c in clauses_by_id.items() if cid.startswith(f"{ref}.")]
    return matches or None


def verify_evidence(
    question: str,
    reranked: list[tuple[SearchHit, float]],
    clauses_by_id: dict[str, Clause],
    min_relevance_score: float,
    min_support_score: float,
) -> EvidenceAssessment:
    relevant = [(hit, score) for hit, score in reranked if score >= min_relevance_score]

    if not relevant:
        return EvidenceAssessment(
            sufficient=False,
            items=[],
            supporting=[],
            reason=RefusalReason.INSUFFICIENT_EVIDENCE,
            detail="No clause in the manual scored above the relevance threshold "
            "for this question.",
        )

    items: list[EvidenceItem] = []
    for hit, relevance_score in relevant:
        unresolved, notes = _check_deferral(question, hit.clause, clauses_by_id)
        support_score = relevance_score * (0.25 if unresolved else 1.0)
        supports = (not unresolved) and support_score >= min_support_score
        items.append(
            EvidenceItem(
                hit=hit,
                relevance_score=relevance_score,
                support_score=support_score,
                supports_answer=supports,
                deferred_unresolved=unresolved,
                notes=notes,
            )
        )

    supporting = [it for it in items if it.supports_answer]

    # If the single most relevant clause found for this question is itself
    # the one flagged as an unresolved deferral, that's a strong signal
    # this question has hit the "apparent gap" failure mode: the clause
    # that looks most on-topic doesn't actually establish the answer.
    # Lower-relevance leftovers shouldn't be allowed to quietly become the
    # answer's citation in that case unless they are independently almost
    # as strong a match as the disqualified top clause -- otherwise the
    # system just answers from the second-best guess instead of admitting
    # the best-looking evidence didn't hold up.
    top_relevance = max(it.relevance_score for it in items) if items else 0.0
    has_deferred_top_clause = any(
        it.deferred_unresolved and it.relevance_score >= 0.85 * top_relevance for it in items
    )
    if has_deferred_top_clause and supporting:
        max_deferred_rel = max(
            it.relevance_score for it in items if it.deferred_unresolved
        )
        supporting = [
            it for it in supporting if it.relevance_score > 1.15 * max_deferred_rel
        ]

    # A clause scoring far below the strongest supporting clause is
    # unlikely to be something the answer should actually cite alongside
    # it -- it clears the floor, but only because the floor has to stay
    # low enough to admit genuine multi-clause answers (see DECISIONS.md).
    # Keep it only if it's within a reasonable margin of the top score.
    if len(supporting) > 1:
        top_score = max(it.support_score for it in supporting)
        supporting = [
            it for it in supporting if it.support_score >= 0.7 * top_score
        ]

    if supporting:
        return EvidenceAssessment(
            sufficient=True, items=items, supporting=supporting, reason=None
        )

    # Relevant evidence exists, but nothing clears the support bar. Decide
    # whether that's an "apparent gap" (ambiguous / deferred) or plain
    # insufficiency.
    if any(it.deferred_unresolved for it in items):
        gap_notes = [n for it in items for n in it.notes]
        return EvidenceAssessment(
            sufficient=False,
            items=items,
            supporting=[],
            reason=RefusalReason.AMBIGUOUS_POLICY,
            detail=" ".join(gap_notes[:2]) if gap_notes else None,
        )

    return EvidenceAssessment(
        sufficient=False,
        items=items,
        supporting=[],
        reason=RefusalReason.AMBIGUOUS_POLICY,
        detail="The manual contains related content, but no retrieved clause "
        "clears the support threshold for a confident answer.",
    )
