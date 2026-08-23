"""
Clause-aware chunking of the Calder County policy manual.

The manual's own numbering (§Part.Section.Paragraph, e.g. §4.3.2) is a
structural feature of the source document, not something we invent -- so
the chunker's only job is to recognise that structure faithfully and never
let a clause's identity get separated from its text.

Design choice: one chunk per clause.
No clause in this ~20-page manual runs long enough to need internal
splitting (the longest is a handful of lines with lettered sub-items), so
we deliberately do not implement paragraph-splitting. If a future revision
of the manual introduced a much longer clause, `_split_if_needed` is the
seam where that would go -- every split fragment keeps the parent clause_id,
per the project brief.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PART_RE = re.compile(r"^#\s*Part\s+(\d+)\s*[—-]\s*(.+?)\s*$")
SECTION_RE = re.compile(r"^##\s*(\d+\.\d+)\s+(.+?)\s*$")
# Two clause-heading styles appear in the manual:
#   "**4.3.2** A recipient must report..."          (plain numbered clause)
#   "**1.4.3 Household** — the applicant or..."      (defined-term clause,
#                                                      term name inside bold)
# The optional group captures the defined term so it isn't lost, then it is
# re-attached to the clause text below.
CLAUSE_RE = re.compile(r"^\*\*(\d+\.\d+\.\d+)(?:\s+([^*]+?))?\*\*\s*(.*)$")
HR_RE = re.compile(r"^-{3,}\s*$")

# Safety valve mentioned in the module docstring. Not exercised by the
# current manual, but present so ingestion degrades gracefully rather than
# silently truncating a future clause that grows past this length.
MAX_CHUNK_CHARS = 2000


@dataclass
class Clause:
    clause_id: str  # e.g. "4.3.2"
    citation: str  # e.g. "§4.3.2"
    part_number: str  # e.g. "4"
    part_title: str  # e.g. "Exclusions"
    section_number: str  # e.g. "4.3"
    section_title: str  # e.g. "Recipient obligations"
    text: str  # full clause text, sub-items and any inline table included
    order: int  # position in document, for stable tie-breaking
    line_start: int
    line_end: int
    # --- Day-2 temporal metadata (all defaults for backward compat) -------
    source: str = "policy-manual.md"
    effective_from: str | None = None   # ISO date, None = always in effect
    effective_to: str | None = None     # ISO date, None = still in effect
    amendment_id: str | None = None     # e.g. "2026-01"
    amendment_paragraph: str | None = None  # e.g. "2.1"
    is_amended_version: bool = False    # True for amended replacement clauses
    date_condition_type: str | None = None  # "determination_date" or "change_date"

    @property
    def full_reference(self) -> str:
        return f"§{self.clause_id} ({self.section_title})"

    def to_metadata(self) -> dict:
        meta = {
            "source": self.source,
            "clause": self.citation,
            "clause_id": self.clause_id,
            "section": self.section_number,
            "section_title": self.section_title,
            "part": f"Part {self.part_number} — {self.part_title}",
        }
        if self.amendment_id:
            meta["amendment_id"] = self.amendment_id
        if self.effective_from:
            meta["effective_from"] = self.effective_from
        if self.effective_to:
            meta["effective_to"] = self.effective_to
        return meta


def parse_clauses(raw_text: str) -> list[Clause]:
    """Parse the manual into a list of Clause objects.

    Walks the document line by line, tracking the current Part and Section
    headings, and accumulates lines into the current clause until a new
    clause marker, a new heading, or a horizontal rule ends it.
    """
    lines = raw_text.splitlines()

    clauses: list[Clause] = []

    current_part_number = ""
    current_part_title = ""
    current_section_number = ""
    current_section_title = ""

    pending: dict | None = None  # in-progress clause being accumulated
    order = 0

    def flush():
        nonlocal pending
        if pending is not None:
            text = "\n".join(pending["lines"]).strip()
            if text:
                clauses.append(
                    Clause(
                        clause_id=pending["clause_id"],
                        citation=f"§{pending['clause_id']}",
                        part_number=pending["part_number"],
                        part_title=pending["part_title"],
                        section_number=pending["section_number"],
                        section_title=pending["section_title"],
                        text=text,
                        order=pending["order"],
                        line_start=pending["line_start"],
                        line_end=pending["line_end"],
                    )
                )
            pending = None

    for i, raw_line in enumerate(lines):
        line = raw_line.rstrip()

        part_match = PART_RE.match(line)
        if part_match:
            flush()
            current_part_number = part_match.group(1)
            current_part_title = part_match.group(2)
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            flush()
            current_section_number = section_match.group(1)
            current_section_title = section_match.group(2)
            continue

        if HR_RE.match(line):
            flush()
            continue

        clause_match = CLAUSE_RE.match(line)
        if clause_match:
            flush()
            order += 1
            term = clause_match.group(2)
            rest = clause_match.group(3) or ""
            first_line = f"{term} {rest}".strip() if term else rest
            pending = {
                "clause_id": clause_match.group(1),
                "part_number": current_part_number,
                "part_title": current_part_title,
                "section_number": current_section_number,
                "section_title": current_section_title,
                "lines": [first_line] if first_line else [],
                "order": order,
                "line_start": i + 1,
                "line_end": i + 1,
            }
            continue

        if pending is not None:
            # Continuation line: sub-items "(a) ...", table rows, wrapped
            # sentences, or a blank line acting as a paragraph break within
            # the same clause. We keep accumulating until the next clause
            # marker/heading/rule -- a blank line alone does not end a
            # clause, since several clauses (e.g. §7.2.1) contain a
            # Markdown table with blank lines around it.
            pending["lines"].append(line)
            pending["line_end"] = i + 1
            continue

        # Lines before the first clause marker (title page, front matter)
        # are intentionally dropped -- they carry no citable clause id and
        # are not policy content a caseworker would be pointed to.

    flush()
    return _split_oversized(clauses)


def _split_oversized(clauses: list[Clause]) -> list[Clause]:
    """Split any clause whose text exceeds MAX_CHUNK_CHARS.

    Every fragment keeps the original clause_id and citation, per the
    project requirement that a split must never lose the clause identity.
    Not expected to trigger on the current manual; kept for robustness.
    """
    result: list[Clause] = []
    for c in clauses:
        if len(c.text) <= MAX_CHUNK_CHARS:
            result.append(c)
            continue
        # Split on paragraph boundaries, keeping every fragment tagged with
        # the same clause_id/citation.
        parts = c.text.split("\n\n")
        buf = ""
        frag_index = 0
        for p in parts:
            if buf and len(buf) + len(p) + 2 > MAX_CHUNK_CHARS:
                frag_index += 1
                result.append(_fragment(c, buf, frag_index))
                buf = p
            else:
                buf = f"{buf}\n\n{p}" if buf else p
        if buf:
            frag_index += 1
            result.append(_fragment(c, buf, frag_index))
    return result


def _fragment(c: Clause, text: str, frag_index: int) -> Clause:
    return Clause(
        clause_id=c.clause_id,
        citation=c.citation,
        part_number=c.part_number,
        part_title=c.part_title,
        section_number=c.section_number,
        section_title=c.section_title,
        text=text.strip(),
        order=c.order,
        line_start=c.line_start,
        line_end=c.line_end,
    )
