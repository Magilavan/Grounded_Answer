"""
Temporal Policy Resolution — the Day-2 core.

Given:
  - A DateContext (from the user's question)
  - Retrieved hits
  - Full corpus of clauses (originals + amended + transitional)

Determines which policy version applies and selects/substitutes the appropriate
clause versions.

Rules encoded:
  - §5.1: Determination-date rules (earnings disregard §6.4.1, income thresholds §6.6.1,
    sanction reduction §10.5.2, sanction exception §10.5.3A) apply to determinations
    made on or after 1 March 2026.
  - §5.2: Change-date rules (reporting deadlines §4.3.2, §9.1.4) apply ONLY to changes
    of circumstances occurring on or after 1 March 2026.
  - §5.3: Claim spanning 1 March 2026 requires apportionment under §7.4.3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from app.ingestion.amendment_parser import ALL_AMENDMENTS, AmendmentSpec
from app.ingestion.chunker import Clause
from app.reasoning.date_extraction import DateContext
from app.retrieval.vector_search import SearchHit

logger = logging.getLogger("grounded_answer.temporal")

EFFECTIVE_DATE = date(2026, 3, 1)

AMENDED_CLAUSE_IDS = {"4.3.2", "9.1.4", "6.4.1", "6.6.1", "10.5.2", "10.5.3A"}
REPORTING_CLAUSE_IDS = {"4.3.2", "9.1.4"}


@dataclass
class TemporalResolution:
    applicable_hits: list[SearchHit]
    excluded_hits: list[SearchHit] = field(default_factory=list)
    date_required_but_missing: bool = False
    missing_date_type: str | None = None
    notes: list[str] = field(default_factory=list)
    transitional_citations: list[str] = field(default_factory=list)


def resolve_temporal_policy(
    hits: list[SearchHit],
    date_ctx: DateContext,
    all_clauses: list[Clause],
    amendments: list[AmendmentSpec] | None = None,
) -> TemporalResolution:
    """Filter and substitute retrieved hits according to the DateContext."""
    result = TemporalResolution(applicable_hits=[])
    clauses_by_id_and_amended: dict[tuple[str, bool], Clause] = {}
    for c in all_clauses:
        clauses_by_id_and_amended[(c.clause_id, c.is_amended_version)] = c

    hit_clause_ids = list(dict.fromkeys([h.clause.clause_id for h in hits]))

    # Check missing date context
    if not date_ctx.has_any_date:
        for cid in hit_clause_ids:
            if cid in AMENDED_CLAUSE_IDS:
                result.date_required_but_missing = True
                if cid in REPORTING_CLAUSE_IDS:
                    result.missing_date_type = "change_date"
                    result.notes.append(
                        "The reporting period depends on when the change of circumstances occurred."
                    )
                else:
                    result.missing_date_type = "determination_date"
                    result.notes.append(
                        f"§{cid} was amended by Amendment No. 2026-01 effective 1 March 2026. "
                        "Applicability depends on determination date."
                    )
                break

    processed_cids: set[str] = set()

    for hit in hits:
        cid = hit.clause.clause_id

        # Transitional provision clauses (e.g. A2026-01-5.1) -> keep as reference
        if cid.startswith("A") and "-" in cid:
            result.applicable_hits.append(hit)
            continue

        if cid in processed_cids:
            continue
        processed_cids.add(cid)

        orig_clause = clauses_by_id_and_amended.get((cid, False))
        amended_clause = clauses_by_id_and_amended.get((cid, True))

        if not amended_clause:
            result.applicable_hits.append(hit)
            continue

        is_reporting = cid in REPORTING_CLAUSE_IDS

        if is_reporting:
            relevant_date = date_ctx.change_date
            condition_type = "change_date"
            trans_cit = "Amendment No. 2026-01 §5.2"
        else:
            relevant_date = (
                date_ctx.determination_date
                or date_ctx.change_date
                or date_ctx.claim_date
                or date_ctx.period_start
            )
            condition_type = "determination_date"
            trans_cit = "Amendment No. 2026-01 §5.1"

        if relevant_date is not None:
            use_amended = relevant_date >= EFFECTIVE_DATE
        else:
            use_amended = False

        if use_amended:
            chosen_clause = amended_clause
            if orig_clause:
                result.excluded_hits.append(
                    SearchHit(clause=orig_clause, score=hit.score, retriever=hit.retriever)
                )
            result.transitional_citations.append(trans_cit)
            result.notes.append(
                f"Using post-amendment version of §{cid} (effective 1 March 2026)."
            )
        else:
            chosen_clause = orig_clause or hit.clause
            if amended_clause:
                result.excluded_hits.append(
                    SearchHit(clause=amended_clause, score=hit.score, retriever=hit.retriever)
                )
            if relevant_date is not None:
                result.transitional_citations.append(trans_cit)
                result.notes.append(
                    f"Using pre-amendment version of §{cid} (for date {relevant_date})."
                )

        # For pre-amendment reporting questions with date context, if §4.3.2 is present, suppress §9.1.4
        if relevant_date is not None and not use_amended and is_reporting and cid == "9.1.4" and "4.3.2" in hit_clause_ids:
            continue

        result.applicable_hits.append(
            SearchHit(clause=chosen_clause, score=hit.score, retriever=hit.retriever)
        )

    # Check for period spanning boundary date (1 March 2026)
    if date_ctx.spans_boundary:
        result.notes.append(
            "Claim period spans 1 March 2026. Under Amendment No. 2026-01 §5.3, "
            "the applicable figures are those in force on each day of the period, "
            "and the award is apportioned accordingly under §7.4.3."
        )
        result.transitional_citations.append("Amendment No. 2026-01 §5.3")

    # Sort applicable_hits so substantive policy clauses rank BEFORE transitional meta-clauses
    def _hit_sort_key(h: SearchHit) -> tuple[int, float]:
        is_transitional = h.clause.clause_id.startswith("A") and "-" in h.clause.clause_id
        return (1 if is_transitional else 0, -h.score)

    result.applicable_hits.sort(key=_hit_sort_key)

    seen_cit: set[str] = set()
    dedup_cits: list[str] = []
    for c in result.transitional_citations:
        if c not in seen_cit:
            seen_cit.add(c)
            dedup_cits.append(c)
    result.transitional_citations = dedup_cits

    return result
