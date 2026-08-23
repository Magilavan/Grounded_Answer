"""
Refusal categories and the user-facing messages for each.

Kept as an enum + template lookup, separate from the decision logic that
chooses among them (decision.py) and separate from evidence verification
(evidence.py) that supplies the reasons -- so that wording can change
(day-two requirement?) without touching the logic that decides *when* to
refuse.
"""

from __future__ import annotations

from enum import Enum


class RefusalReason(str, Enum):
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS_POLICY = "AMBIGUOUS_POLICY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    CONFLICTING_POLICY = "CONFLICTING_POLICY"
    CITATION_UNVERIFIABLE = "CITATION_UNVERIFIABLE"
    MISSING_DATE_CONTEXT = "MISSING_DATE_CONTEXT"


ESCALATION_LINE = (
    "Please consult the appropriate policy administrator or benefits supervisor."
)

_TEMPLATES: dict[RefusalReason, str] = {
    RefusalReason.INSUFFICIENT_EVIDENCE: (
        "I can't determine this from the policy manual.\n\n"
        "The manual does not provide enough information to answer this "
        "question confidently.\n\n" + ESCALATION_LINE
    ),
    RefusalReason.AMBIGUOUS_POLICY: (
        "The policy manual contains related information, but it does not "
        "clearly establish the answer to this question.\n\n"
        "I cannot determine the correct interpretation from the manual "
        "alone.\n\n" + ESCALATION_LINE
    ),
    RefusalReason.OUT_OF_SCOPE: (
        "This question falls outside the Calder County Household Support "
        "Program policy manual, which is the only source this assistant is "
        "authorised to answer from.\n\n" + ESCALATION_LINE
    ),
    RefusalReason.CITATION_UNVERIFIABLE: (
        "I found a candidate answer, but could not verify its citation "
        "against the retrieved evidence, so I'm not returning it as a "
        "grounded answer.\n\n" + ESCALATION_LINE
    ),
    RefusalReason.MISSING_DATE_CONTEXT: (
        "The applicable reporting period depends on when the change of circumstances occurred.\n\n"
        "Please provide the relevant date so I can determine which policy provision applies.\n\n" + ESCALATION_LINE
    ),
}


def refusal_message(reason: RefusalReason, detail: str | None = None) -> str:
    base = _TEMPLATES.get(reason, _TEMPLATES[RefusalReason.INSUFFICIENT_EVIDENCE])
    if detail and reason != RefusalReason.MISSING_DATE_CONTEXT:
        return f"{base}\n\n({detail})"
    return base
