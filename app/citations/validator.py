"""
Citation validation.

Every clause id that ends up in a final answer must pass through here.
This is what makes "never let the LLM invent citations" (brief section 17)
an enforced property of the system rather than a hopeful line in a prompt:
even if an optional LLM generation step is used, its output is only
accepted after every citation it contains is checked against this
validator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.chunker import Clause

CITATION_RE = re.compile(
    r"(?:Amendment\s+No\.\s*[\d-]+\s+)?§\s*(\d+(?:\.\d+)+(?:[A-Z])?|\d+\.\d+)"
)


@dataclass
class ValidationResult:
    valid: bool
    checked_citations: list[str]
    invalid_citations: list[str]
    unsupported_citations: list[str]  # exists in manual, but wasn't in the evidence used
    issues: list[str]


def extract_citations(text: str) -> list[str]:
    # Extracts raw clause references like "4.3.2", "10.5.3A", "5.2", etc.
    return CITATION_RE.findall(text)


def validate_citations(
    text: str,
    supporting_clauses: list[Clause],
    all_clauses_by_id: dict[str, Clause],
) -> ValidationResult:
    """Validate every §X.X.X or amendment citation found in `text`.

    A citation is valid only if:
      1. It exists in the manual or amendments (not hallucinated), and
      2. It corresponds to a clause that was actually part of the
         evidence used to build this answer (`supporting_clauses`).
    """
    cited = extract_citations(text)
    supporting_ids = set()
    for c in supporting_clauses:
        supporting_ids.add(c.clause_id)
        supporting_ids.add(c.citation)
        if c.amendment_paragraph:
            supporting_ids.add(c.amendment_paragraph)
            supporting_ids.add(f"5.{c.amendment_paragraph}")

    # Build lookup of all valid clause IDs / citations / section refs across corpus
    valid_known = set(all_clauses_by_id.keys())
    for cid, c in all_clauses_by_id.items():
        valid_known.add(c.citation)
        valid_known.add(c.section_number)
        if c.amendment_paragraph:
            valid_known.add(c.amendment_paragraph)

    invalid: list[str] = []
    unsupported: list[str] = []
    issues: list[str] = []

    for cid in cited:
        # Match against cid or clause_id or section
        if cid not in valid_known and not any(k.startswith(f"{cid}.") for k in all_clauses_by_id):
            invalid.append(cid)
            issues.append(f"§{cid} does not exist in the policy manual or amendments.")
        elif cid not in supporting_ids and not any(sc.clause_id == cid or sc.section_number == cid or (sc.amendment_paragraph and sc.amendment_paragraph == cid) for sc in supporting_clauses):
            unsupported.append(cid)
            issues.append(
                f"§{cid} exists in the manual but was not part of the "
                "evidence retrieved and verified for this answer."
            )

    valid = not invalid and not unsupported
    return ValidationResult(
        valid=valid,
        checked_citations=cited,
        invalid_citations=invalid,
        unsupported_citations=unsupported,
        issues=issues,
    )
