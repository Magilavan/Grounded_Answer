"""
Extract dates and temporal context from natural-language policy questions.

The system must understand WHAT date the question refers to (change-of-
circumstances date, determination date, claim period, etc.) because
different amendment transitional provisions key on different date types.

Uses regex-based parsing -- no external NLP dependency needed for the date
formats that appear in policy questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date


# ---------------------------------------------------------------------------
# DateContext
# ---------------------------------------------------------------------------

@dataclass
class DateContext:
    """Structured temporal context extracted from a user question."""
    claim_date: date | None = None
    change_date: date | None = None
    determination_date: date | None = None
    period_start: date | None = None
    period_end: date | None = None
    raw_date_text: str | None = None
    # Whether the question explicitly names a date type
    date_type_explicit: bool = False
    # Whether the question involves a period spanning a boundary
    spans_boundary: bool = False
    cleaned_question: str = ""

    @property
    def has_any_date(self) -> bool:
        return any([
            self.claim_date, self.change_date, self.determination_date,
            self.period_start, self.period_end,
        ])

    @property
    def primary_date(self) -> date | None:
        """The most relevant date for temporal resolution."""
        return (
            self.change_date
            or self.determination_date
            or self.claim_date
            or self.period_start
        )


# ---------------------------------------------------------------------------
# Month name mapping
# ---------------------------------------------------------------------------

MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _parse_month(s: str) -> int | None:
    return MONTH_NAMES.get(s.lower().rstrip("."))


# ---------------------------------------------------------------------------
# Date extraction patterns
# ---------------------------------------------------------------------------

# "20 February 2026", "February 20, 2026", "February 20th, 2026"
_DMY = re.compile(
    r"\b(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"(?:\s*,?\s*|\s+)(\d{4})\b",
    re.IGNORECASE,
)
_MDY = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:\s*,?\s*|\s+)(\d{4})\b",
    re.IGNORECASE,
)

# "Feb 20 2026", "Feb 2026"
_SHORT_MDY = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:\s*,?\s*|\s+)(\d{4})\b",
    re.IGNORECASE,
)
_SHORT_MY = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)

# "February 2026" (month + year, no day)
_MONTH_YEAR = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)

# ISO-style: "2026-03-01"
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# DD/MM/YYYY or MM/DD/YYYY -- assume DD/MM for day > 12
_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")

# "March 1" without year (assume 2026 for policy context)
_MONTH_DAY_NOYEAR = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?\b"
    r"(?!\s*,?\s*\d{4})",
    re.IGNORECASE,
)


def _extract_raw_dates(text: str) -> list[date]:
    """Extract all date mentions from text, returning parsed date objects."""
    dates: list[date] = []

    for m in _ISO.finditer(text):
        try:
            dates.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass

    for m in _DMY.finditer(text):
        month = _parse_month(m.group(2))
        if month:
            try:
                dates.append(date(int(m.group(3)), month, int(m.group(1))))
            except ValueError:
                pass

    for m in _MDY.finditer(text):
        month = _parse_month(m.group(1))
        if month:
            try:
                dates.append(date(int(m.group(3)), month, int(m.group(2))))
            except ValueError:
                pass

    for m in _SHORT_MDY.finditer(text):
        month = _parse_month(m.group(1))
        if month:
            try:
                dates.append(date(int(m.group(3)), month, int(m.group(2))))
            except ValueError:
                pass

    for m in _SLASH.finditer(text):
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            if a > 12:
                dates.append(date(y, b, a))  # DD/MM/YYYY
            else:
                dates.append(date(y, a, b))  # MM/DD/YYYY
        except ValueError:
            pass

    # Month+year without day -> use 15th as midpoint
    for m in _MONTH_YEAR.finditer(text):
        month = _parse_month(m.group(1))
        if month:
            try:
                dates.append(date(int(m.group(2)), month, 15))
            except ValueError:
                pass

    for m in _SHORT_MY.finditer(text):
        month = _parse_month(m.group(1))
        if month:
            try:
                dates.append(date(int(m.group(2)), month, 15))
            except ValueError:
                pass

    # "March 1" without year -> assume 2026
    for m in _MONTH_DAY_NOYEAR.finditer(text):
        month = _parse_month(m.group(1))
        if month:
            try:
                dates.append(date(2026, month, int(m.group(2))))
            except ValueError:
                pass

    # Deduplicate preserving order
    seen: set[date] = set()
    unique: list[date] = []
    for d in dates:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


# ---------------------------------------------------------------------------
# Date type classification
# ---------------------------------------------------------------------------

# Patterns that indicate the date refers to a change of circumstances
_CHANGE_DATE_INDICATORS = [
    r"change[sd]?\s+(on|in|of|occurring)",
    r"(income|circumstances?)\s+(changed|change[sd]?)\s+(on|in)",
    r"report\w*\s+(a|the)?\s*change",
    r"change\s+of\s+circumstances",
    r"change\s+occur",
    r"reported?\s+(a|the)?\s*change",
    r"reporting\s+(deadline|period|requirement)",
    r"how\s+(many|long)\s+.*days?\s+.*report",
]

_DETERMINATION_DATE_INDICATORS = [
    r"determination\s+(made|on|in|dated)",
    r"(assessed|decided|determined)\s+(on|in)",
    r"disregard\s+(for|in|on)",
    r"earnings?\s+disregard",
    r"income\s+threshold",
    r"sanction",
]

_PERIOD_INDICATORS = [
    r"(claim|period)\s+(cover|span|from|running)",
    r"(from|between)\s+.*\s+(to|through|and)\s+",
    r"spanning",
]


def _classify_date_type(text: str) -> str:
    """Classify what type of date the question is asking about."""
    lowered = text.lower()

    for pattern in _CHANGE_DATE_INDICATORS:
        if re.search(pattern, lowered):
            return "change_date"

    for pattern in _DETERMINATION_DATE_INDICATORS:
        if re.search(pattern, lowered):
            return "determination_date"

    for pattern in _PERIOD_INDICATORS:
        if re.search(pattern, lowered):
            return "period"

    return "unspecified"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DATE_PAT = r"(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?,?\s+\d{4}|\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})"

_DATE_PHRASE_RE = re.compile(
    r"\b(?:occurring\s+on|made\s+on|dated|as\s+of|running\s+from|from|through)\s+" + _DATE_PAT + r"\b|\b(?:on)\s+" + _DATE_PAT + r"\b|\b" + _DATE_PAT + r"\b",
    re.IGNORECASE,
)


def _clean_question_dates(question: str) -> str:
    cleaned = _DATE_PHRASE_RE.sub("", question)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r",\s*\?", "?", cleaned)
    cleaned = re.sub(r"\bfor\s*\?", "?", cleaned)
    cleaned = re.sub(r"\bfor\s*,", "", cleaned)
    return cleaned.strip()


def extract_date_context(question: str) -> DateContext:
    """Extract a DateContext from a natural-language policy question.

    Identifies dates, classifies them by type (change date, determination
    date, claim period), and detects if the question involves a period
    spanning a policy boundary date.
    """
    dates = _extract_raw_dates(question)
    date_type = _classify_date_type(question)
    cleaned_q = _clean_question_dates(question) if dates else question
    ctx = DateContext(raw_date_text=question, cleaned_question=cleaned_q)

    if not dates:
        return ctx

    ctx.date_type_explicit = date_type != "unspecified"

    if date_type == "change_date":
        ctx.change_date = dates[0]
    elif date_type == "determination_date":
        ctx.determination_date = dates[0]
    elif date_type == "period":
        ctx.period_start = dates[0]
        if len(dates) >= 2:
            ctx.period_end = dates[1]
            # Check if period spans March 1, 2026
            boundary = date(2026, 3, 1)
            if ctx.period_start < boundary <= ctx.period_end:
                ctx.spans_boundary = True
    else:
        # Unspecified: try to infer from question context
        # If the question is about reporting, assume change_date
        if re.search(r"report|reporting|days?\s+.*report", question, re.IGNORECASE):
            ctx.change_date = dates[0]
        else:
            # Default to determination_date for other questions
            ctx.determination_date = dates[0]

    return ctx
