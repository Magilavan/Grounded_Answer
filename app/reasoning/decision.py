"""
The single place that decides ANSWER / REFUSE / CONFLICT.

Nothing upstream (retrieval, reranking) and nothing downstream (generation,
citation formatting) makes this call -- it happens here, once, so that it
can be tested, logged, and changed independently of everything else. See
section 15 of the project brief and DECISIONS.md ("where the line between
answering and refusing was set").

Order of operations:
  1. Evidence verification runs first and determines the relevant
     candidate pool (clauses clearing the relevance floor).
  2. Contradiction detection runs over that same candidate pool. A
     detected contradiction wins over an otherwise-sufficient answer --
     the brief is explicit that a conflict must be surfaced, never
     silently resolved by picking one side.
  3. Only if there is no contradiction does evidence sufficiency decide
     ANSWER vs REFUSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import re

from app.ingestion.chunker import Clause
from app.reasoning.contradiction_detector import Contradiction, detect_contradictions
from app.reasoning.evidence import EvidenceAssessment, verify_evidence
from app.reasoning.refusal import RefusalReason
from app.retrieval.vector_search import SearchHit

# Matches both a full clause reference ("§4.3.2") and a section-level
# reference ("§4.3") -- the manual uses both forms when one clause points
# at another.
_CLAUSE_REF_RE = re.compile(r"§\s*(\d+\.\d+(?:\.\d+)?)")


class DecisionType(str, Enum):
    ANSWER = "ANSWER"
    REFUSE = "REFUSE"
    CONFLICT = "CONFLICT"


@dataclass
class Decision:
    decision_type: DecisionType
    evidence: EvidenceAssessment
    contradictions: list[Contradiction]
    refusal_reason: RefusalReason | None = None


def decide(
    question: str,
    reranked: list[tuple[SearchHit, float]],
    clauses_by_id: dict[str, Clause],
    min_relevance_score: float,
    min_support_score: float,
    excluded_clause_ids: set[str] | None = None,
) -> Decision:
    evidence = verify_evidence(
        question=question,
        reranked=reranked,
        clauses_by_id=clauses_by_id,
        min_relevance_score=min_relevance_score,
        min_support_score=min_support_score,
    )

    candidate_clauses = [it.hit.clause for it in evidence.items]

    # A clause that explicitly cross-references another clause for the
    # same fact (e.g. "...required under §4.3") is pulled into the
    # contradiction-checking pool even if retrieval didn't independently
    # surface it. Otherwise a contradiction where only one side of the
    # conflict was lexically similar enough to the question to be
    # retrieved would never be detected -- exactly the situation where
    # the referencing clause misstates the value the target clause
    # actually establishes.
    excluded_set = excluded_clause_ids or set()
    seen_ids = {c.clause_id for c in candidate_clauses}
    for clause in list(candidate_clauses):
        for ref_id in _CLAUSE_REF_RE.findall(clause.text):
            if ref_id in seen_ids or ref_id in excluded_set:
                continue
            if ref_id in clauses_by_id:
                # Full clause reference, e.g. "§9.1.4".
                if ref_id not in excluded_set:
                    candidate_clauses.append(clauses_by_id[ref_id])
                    seen_ids.add(ref_id)
            else:
                # Section-level reference, e.g. "§4.3" -- pull in every
                # clause under that section, since the referencing clause
                # is making a claim about the section as a whole.
                for cid, c in clauses_by_id.items():
                    if cid.startswith(f"{ref_id}.") and cid not in seen_ids and cid not in excluded_set:
                        candidate_clauses.append(c)
                        seen_ids.add(cid)

    contradictions = (
        detect_contradictions(candidate_clauses, clauses_by_id) if candidate_clauses else []
    )

    if contradictions:
        return Decision(
            decision_type=DecisionType.CONFLICT,
            evidence=evidence,
            contradictions=contradictions,
            refusal_reason=RefusalReason.CONFLICTING_POLICY,
        )

    if evidence.sufficient:
        return Decision(
            decision_type=DecisionType.ANSWER,
            evidence=evidence,
            contradictions=[],
            refusal_reason=None,
        )

    return Decision(
        decision_type=DecisionType.REFUSE,
        evidence=evidence,
        contradictions=[],
        refusal_reason=evidence.reason or RefusalReason.INSUFFICIENT_EVIDENCE,
    )
