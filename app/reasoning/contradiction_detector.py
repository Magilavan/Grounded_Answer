"""
Detect whether two or more retrieved clauses establish incompatible rules
about the same fact.

The mechanism is deliberately narrow and numeric rather than a general
semantic-conflict detector, because a broad heuristic over a 148-clause
corpus produces false positives fast (see DECISIONS.md for two near-misses
this was tuned against: §3.2.1/§3.2.2 and §5.2.1/§5.2.2, which state a
period and its *explicit, signalled* extension -- not a conflict -- versus
§4.3.2/§9.1.4, which state two different, unsignalled values for what is
presented as the same reporting obligation).

Heuristic:
  1. Two clauses are "about the same fact" if their content-word overlap
     (Jaccard, stopwords/numbers excluded) clears a threshold.
  2. Both must contain a number tied to a comparable unit (days, weeks,
     months, years, per cent).
  3. If the numbers differ, and neither clause's number is introduced by an
     explicit modifier phrase ("extended to", "increased to", "reduced to",
     "instead of", "in place of") that signals a deliberate, acknowledged
     change rather than a silent inconsistency, it's flagged as a conflict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

from app.ingestion.chunker import Clause
from app.reasoning.evidence import _content_tokens

UNIT_NUMBER_RE = re.compile(
    r"(\d[\d,]*)\s*(?:calendar\s+|business\s+|working\s+)?"
    r"(day|days|week|weeks|month|months|year|years|per\s*cent|percent|%)",
    re.IGNORECASE,
)

MODIFIER_WINDOW = 25  # chars to look back from a number for a modifier phrase
MODIFIER_PHRASES = (
    "extended to",
    "extended when",
    "increased to",
    "increases to",
    "increase to",
    "reduced to",
    "reduces to",
    "in place of",
    "instead of",
    "rather than",
)

MIN_OVERLAP = 0.40

# Phrases that mean "the number I just stated is what that other clause
# requires" -- i.e. an explicit, checkable assertion about another
# clause's content, as opposed to an ordinary informative cross-reference.
# This is what lets the detector catch §9.1.4 ("...required under §4.3")
# vs §4.3.2 without also flagging every pair of clauses in the corpus that
# happen to both use the words "within" and "days" for entirely unrelated
# deadlines (application review, appeals, panel hearings, etc.) -- see
# DECISIONS.md for the false-positive sweep this was tuned against.
ASSERTION_REF_RE = re.compile(
    r"(?:required|permitted|specified|provided|set out|established)\s+under\s+"
    r"§\s*(\d+\.\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _unit_family(unit: str) -> str:
    unit = unit.lower().replace(" ", "")
    if unit.startswith("day"):
        return "day"
    if unit.startswith("week"):
        return "week"
    if unit.startswith("month"):
        return "month"
    if unit.startswith("year"):
        return "year"
    if unit in ("percent", "%", "percent."):
        return "percent"
    return unit


@dataclass
class NumericClaim:
    value: str
    unit: str
    has_modifier: bool
    context: str


def _extract_claims(text: str) -> list[NumericClaim]:
    claims = []
    for m in UNIT_NUMBER_RE.finditer(text):
        start = max(0, m.start() - MODIFIER_WINDOW)
        preceding = text[start : m.start()].lower()
        has_modifier = any(phrase in preceding for phrase in MODIFIER_PHRASES)
        claims.append(
            NumericClaim(
                value=m.group(1).replace(",", ""),
                unit=_unit_family(m.group(2)),
                has_modifier=has_modifier,
                context=text[max(0, m.start() - 40) : m.end() + 10].strip(),
            )
        )
    return claims


@dataclass
class Contradiction:
    clause_a: Clause
    clause_b: Clause
    unit: str
    value_a: str
    value_b: str
    context_a: str
    context_b: str


def _resolve_targets(target_id: str, clauses_by_id: dict[str, Clause]) -> list[Clause]:
    if target_id in clauses_by_id:
        return [clauses_by_id[target_id]]
    return [c for cid, c in clauses_by_id.items() if cid.startswith(f"{target_id}.")]


def _detect_assertion_contradictions(
    clauses: list[Clause], clauses_by_id: dict[str, Clause]
) -> list[Contradiction]:
    """Catch a clause that explicitly asserts what another clause
    "requires"/"permits"/"specifies" with a number that the referenced
    clause doesn't actually state.

    This is a stronger, more specific signal than generic lexical overlap
    (see module docstring): it only fires when a clause makes a checkable
    claim about another clause's content, so it doesn't confuse two
    clauses that merely share administrative vocabulary ("within",
    "days", "determination") while governing entirely different
    processes.
    """
    findings: list[Contradiction] = []
    for clause in clauses:
        for m in ASSERTION_REF_RE.finditer(clause.text):
            target_id = m.group(1)
            preceding = clause.text[: m.start()]
            preceding_matches = list(UNIT_NUMBER_RE.finditer(preceding))
            if not preceding_matches:
                continue
            asserted = preceding_matches[-1]
            asserted_value = asserted.group(1).replace(",", "")
            asserted_unit = _unit_family(asserted.group(2))
            asserted_context = clause.text[
                max(0, asserted.start() - 30) : m.end() + 5
            ].strip()

            for target in _resolve_targets(target_id, clauses_by_id):
                if target.clause_id == clause.clause_id:
                    continue
                target_claims = [
                    c
                    for c in _extract_claims(target.text)
                    if not c.has_modifier and c.unit == asserted_unit
                ]
                for tc in target_claims:
                    if tc.value != asserted_value:
                        findings.append(
                            Contradiction(
                                clause_a=clause,
                                clause_b=target,
                                unit=asserted_unit,
                                value_a=asserted_value,
                                value_b=tc.value,
                                context_a=asserted_context,
                                context_b=tc.context,
                            )
                        )
    return findings


def detect_contradictions(
    clauses: list[Clause], clauses_by_id: dict[str, Clause] | None = None
) -> list[Contradiction]:
    """Look for numeric contradictions among the given clauses.

    Runs two complementary passes:
      1. Explicit-assertion detection ("...required under §X.Y") -- high
         precision, fires only on a checkable cross-clause claim.
      2. Generic same-topic numeric mismatch, gated by a stricter overlap
         threshold -- a fallback for a contradiction that isn't phrased as
         an explicit cross-reference.

    Intended to be called on the small set of clauses that retrieval
    already judged relevant to the current question (the caller,
    app/reasoning/decision.py, also pulls in any clause one of them
    explicitly cross-references) -- not the whole manual on every query.
    """
    findings: list[Contradiction] = []
    lookup = clauses_by_id or {c.clause_id: c for c in clauses}

    findings.extend(_detect_assertion_contradictions(clauses, lookup))

    token_cache = {c.clause_id: _content_tokens(c.text) for c in clauses}
    seen_pairs = {
        tuple(sorted([f.clause_a.clause_id, f.clause_b.clause_id])) for f in findings
    }

    for a, b in combinations(clauses, 2):
        if a.clause_id == b.clause_id:
            continue
        pair_key = tuple(sorted([a.clause_id, b.clause_id]))
        if pair_key in seen_pairs:
            continue
        tokens_a, tokens_b = token_cache[a.clause_id], token_cache[b.clause_id]
        if not tokens_a or not tokens_b:
            continue
        overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        if overlap < MIN_OVERLAP:
            continue

        claims_a = [c for c in _extract_claims(a.text) if not c.has_modifier]
        claims_b = [c for c in _extract_claims(b.text) if not c.has_modifier]

        for ca in claims_a:
            for cb in claims_b:
                if ca.unit == cb.unit and ca.value != cb.value:
                    findings.append(
                        Contradiction(
                            clause_a=a,
                            clause_b=b,
                            unit=ca.unit,
                            value_a=ca.value,
                            value_b=cb.value,
                            context_a=ca.context,
                            context_b=cb.context,
                        )
                    )
                    seen_pairs.add(pair_key)
    return findings
