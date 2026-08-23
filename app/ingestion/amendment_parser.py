"""
Parse Amendment No. 2026-01 and produce:
  1. Amended versions of existing clauses (with text substitutions applied)
  2. New clauses introduced by the amendment (e.g. §10.5.3A)
  3. Transitional provision clauses (§5.1–§5.3 of the amendment)

The amendment rules are encoded as structured data, not hard-coded answers.
Another amendment can be added later by defining a new AmendmentSpec with its
own rules and transitional provisions, using the same framework.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import date

from app.ingestion.chunker import Clause


# ---------------------------------------------------------------------------
# Data model for amendments
# ---------------------------------------------------------------------------

@dataclass
class SubstitutionRule:
    """One textual substitution within a clause."""
    paragraph: str              # amendment paragraph, e.g. "2.1"
    target_clause_id: str       # clause being modified, e.g. "4.3.2"
    find_text: str              # text to find (case-sensitive substring)
    replace_text: str           # replacement text
    replace_all: bool = True    # replace all occurrences?
    category: str = ""          # "reporting", "earnings_disregard", etc.
    transitional_ref: str = ""  # e.g. "5.1" or "5.2"
    date_condition_type: str = "determination_date"  # or "change_date"


@dataclass
class InsertionRule:
    """A new clause inserted into the manual by an amendment."""
    paragraph: str              # amendment paragraph, e.g. "4.2"
    new_clause_id: str          # e.g. "10.5.3A"
    after_clause_id: str        # clause it follows, e.g. "10.5.3"
    text: str                   # full text of the new clause
    category: str = ""
    transitional_ref: str = ""
    date_condition_type: str = "determination_date"


@dataclass
class TableReplacementRule:
    """Replace the table portion of a clause."""
    paragraph: str
    target_clause_id: str
    new_table: str              # markdown table to substitute
    category: str = ""
    transitional_ref: str = ""
    date_condition_type: str = "determination_date"


@dataclass
class TransitionalProvision:
    """A transitional provision from the amendment, stored as a retrievable clause."""
    provision_id: str           # e.g. "5.1"
    text: str
    applies_to_categories: list[str] = field(default_factory=list)
    date_condition_type: str = "determination_date"


@dataclass
class AmendmentSpec:
    """Complete specification of a single amendment to the policy manual."""
    amendment_id: str           # e.g. "2026-01"
    issued_date: date
    effective_date: date
    source_file: str            # e.g. "Amendment No. 2026-01.md"
    substitutions: list[SubstitutionRule] = field(default_factory=list)
    insertions: list[InsertionRule] = field(default_factory=list)
    table_replacements: list[TableReplacementRule] = field(default_factory=list)
    transitional_provisions: list[TransitionalProvision] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Amendment No. 2026-01
# ---------------------------------------------------------------------------

AMENDMENT_2026_01 = AmendmentSpec(
    amendment_id="2026-01",
    issued_date=date(2026, 2, 12),
    effective_date=date(2026, 3, 1),
    source_file="Amendment No. 2026-01.md",
    substitutions=[
        SubstitutionRule(
            paragraph="1.1",
            target_clause_id="6.4.1",
            find_text="$120 per month",
            replace_text="$175 per month",
            replace_all=True,
            category="earnings_disregard",
            transitional_ref="5.1",
            date_condition_type="determination_date",
        ),
        SubstitutionRule(
            paragraph="2.1",
            target_clause_id="4.3.2",
            find_text="10 calendar days",
            replace_text="14 calendar days",
            replace_all=True,
            category="reporting",
            transitional_ref="5.2",
            date_condition_type="change_date",
        ),
        SubstitutionRule(
            paragraph="2.2",
            target_clause_id="9.1.4",
            find_text="30 calendar days",
            replace_text="14 calendar days",
            replace_all=True,
            category="reporting",
            transitional_ref="5.2",
            date_condition_type="change_date",
        ),
        SubstitutionRule(
            paragraph="4.1",
            target_clause_id="10.5.2",
            find_text="20 per cent",
            replace_text="15 per cent",
            replace_all=True,
            category="sanction",
            transitional_ref="5.1",
            date_condition_type="determination_date",
        ),
    ],
    table_replacements=[
        TableReplacementRule(
            paragraph="3.1",
            target_clause_id="6.6.1",
            new_table=(
                "| Household size | Monthly threshold |\n"
                "|:--|:--|\n"
                "| 1 | $1,225 |\n"
                "| 2 | $1,650 |\n"
                "| 3 | $2,075 |\n"
                "| 4 | $2,500 |\n"
                "| 5 | $2,925 |\n"
                "| each additional member | + $425 |"
            ),
            category="income_threshold",
            transitional_ref="5.1",
            date_condition_type="determination_date",
        ),
    ],
    insertions=[
        InsertionRule(
            paragraph="4.2",
            new_clause_id="10.5.3A",
            after_clause_id="10.5.3",
            text=(
                "A sanction must not be imposed in respect of a failure to "
                "report where the change of circumstances in question would "
                "have increased the award."
            ),
            category="sanction",
            transitional_ref="5.1",
            date_condition_type="determination_date",
        ),
    ],
    transitional_provisions=[
        TransitionalProvision(
            provision_id="5.1",
            text=(
                "The amendments made by paragraphs 1, 3 and 4 apply to any "
                "determination made on or after 1 March 2026, including a "
                "determination in respect of a period before that date."
            ),
            applies_to_categories=[
                "earnings_disregard", "income_threshold", "sanction",
            ],
            date_condition_type="determination_date",
        ),
        TransitionalProvision(
            provision_id="5.2",
            text=(
                "The amendments made by paragraph 2 apply only in respect of "
                "a change of circumstances occurring on or after 1 March 2026. "
                "Where the change of circumstances occurred before 1 March "
                "2026, the reporting period is the period that applied at the "
                "date of the change, irrespective of the date of the "
                "determination."
            ),
            applies_to_categories=["reporting"],
            date_condition_type="change_date",
        ),
        TransitionalProvision(
            provision_id="5.3",
            text=(
                "Where a claim relates to a period spanning 1 March 2026, "
                "the applicable figures are those in force on each day of "
                "the period, and the award is apportioned accordingly under "
                "\u00a77.4.3."
            ),
            applies_to_categories=[
                "earnings_disregard", "income_threshold",
                "sanction", "reporting",
            ],
            date_condition_type="determination_date",
        ),
    ],
)

# Registry: all known amendments in chronological order.
# A future amendment would simply be appended here.
ALL_AMENDMENTS: list[AmendmentSpec] = [AMENDMENT_2026_01]


# ---------------------------------------------------------------------------
# Generate amended clause versions
# ---------------------------------------------------------------------------

_TABLE_RE = re.compile(
    r"\|[^\n]*\|\n(?:\|[^\n]*\|\n?)+",
    re.MULTILINE,
)


def _apply_substitutions(
    original: Clause,
    amendment: AmendmentSpec,
) -> list[Clause]:
    """Create amended clause versions by applying substitution rules."""
    results: list[Clause] = []
    for rule in amendment.substitutions:
        if rule.target_clause_id != original.clause_id:
            continue
        amended = copy.deepcopy(original)
        if rule.replace_all:
            # Handle bold-wrapped variants too (manual uses **10 calendar days**)
            bold_find = f"**{rule.find_text}**"
            if bold_find in amended.text:
                amended.text = amended.text.replace(bold_find, f"**{rule.replace_text}**")
            amended.text = amended.text.replace(rule.find_text, rule.replace_text)
        else:
            amended.text = amended.text.replace(rule.find_text, rule.replace_text, 1)

        amended.source = amendment.source_file
        amended.amendment_id = amendment.amendment_id
        amended.amendment_paragraph = rule.paragraph
        amended.is_amended_version = True
        amended.effective_from = amendment.effective_date.isoformat()
        amended.date_condition_type = rule.date_condition_type
        results.append(amended)
    return results


def _apply_table_replacements(
    original: Clause,
    amendment: AmendmentSpec,
) -> list[Clause]:
    """Create amended clause versions by replacing the table."""
    results: list[Clause] = []
    for rule in amendment.table_replacements:
        if rule.target_clause_id != original.clause_id:
            continue
        amended = copy.deepcopy(original)
        # Replace the markdown table in the clause text
        table_match = _TABLE_RE.search(amended.text)
        if table_match:
            amended.text = (
                amended.text[:table_match.start()]
                + rule.new_table
                + amended.text[table_match.end():]
            )
        else:
            # Fallback: append the new table
            amended.text += "\n\n" + rule.new_table

        amended.source = amendment.source_file
        amended.amendment_id = amendment.amendment_id
        amended.amendment_paragraph = rule.paragraph
        amended.is_amended_version = True
        amended.effective_from = amendment.effective_date.isoformat()
        amended.date_condition_type = rule.date_condition_type
        results.append(amended)
    return results


def _generate_insertions(amendment: AmendmentSpec) -> list[Clause]:
    """Create new Clause objects for clauses inserted by the amendment."""
    results: list[Clause] = []
    for rule in amendment.insertions:
        clause = Clause(
            clause_id=rule.new_clause_id,
            citation=f"§{rule.new_clause_id}",
            part_number=rule.after_clause_id.split(".")[0],
            part_title="",  # filled during ingestion
            section_number=".".join(rule.after_clause_id.split(".")[:2]),
            section_title="",  # filled during ingestion
            text=rule.text,
            order=9000,  # high order to sort after originals
            line_start=0,
            line_end=0,
            source=amendment.source_file,
            amendment_id=amendment.amendment_id,
            amendment_paragraph=rule.paragraph,
            is_amended_version=True,
            effective_from=amendment.effective_date.isoformat(),
            date_condition_type=rule.date_condition_type,
        )
        results.append(clause)
    return results


def _generate_transitional_clauses(amendment: AmendmentSpec) -> list[Clause]:
    """Create retrievable Clause objects for the amendment's
    transitional provisions so the system can cite them."""
    results: list[Clause] = []
    for tp in amendment.transitional_provisions:
        clause_id = f"A{amendment.amendment_id}-{tp.provision_id}"
        clause = Clause(
            clause_id=clause_id,
            citation=f"Amendment No. {amendment.amendment_id} §{tp.provision_id}",
            part_number="0",
            part_title=f"Amendment No. {amendment.amendment_id}",
            section_number="5",
            section_title="Transitional provision",
            text=tp.text,
            order=9100,
            line_start=0,
            line_end=0,
            source=amendment.source_file,
            amendment_id=amendment.amendment_id,
            amendment_paragraph=tp.provision_id,
            is_amended_version=False,
            effective_from=amendment.effective_date.isoformat(),
            date_condition_type=tp.date_condition_type,
        )
        results.append(clause)
    return results


def generate_amendment_clauses(
    original_clauses: list[Clause],
    amendments: list[AmendmentSpec] | None = None,
) -> list[Clause]:
    """Given the original policy clauses and a list of amendments, produce
    all amended clause versions, inserted clauses, and transitional
    provision clauses that should be added to the index.

    The original clauses are also marked with effective_to when they are
    superseded by an amendment rule.
    """
    amendments = amendments or ALL_AMENDMENTS
    originals_by_id: dict[str, Clause] = {c.clause_id: c for c in original_clauses}
    new_clauses: list[Clause] = []

    for amendment in amendments:
        # Track which original clause IDs are affected
        affected_ids: set[str] = set()

        for orig in original_clauses:
            amended = _apply_substitutions(orig, amendment)
            new_clauses.extend(amended)
            if amended:
                affected_ids.add(orig.clause_id)

            table_amended = _apply_table_replacements(orig, amendment)
            new_clauses.extend(table_amended)
            if table_amended:
                affected_ids.add(orig.clause_id)

        # Mark originals as having effective_to for superseded clauses
        cutoff = (amendment.effective_date.toordinal() - 1)
        cutoff_date = date.fromordinal(cutoff).isoformat()
        for cid in affected_ids:
            if cid in originals_by_id:
                originals_by_id[cid].effective_to = cutoff_date

        # Inserted clauses
        insertions = _generate_insertions(amendment)
        # Fill part/section info from the clause they follow
        for ins_rule in amendment.insertions:
            after = originals_by_id.get(ins_rule.after_clause_id)
            if after:
                for ins_clause in insertions:
                    if ins_clause.clause_id == ins_rule.new_clause_id:
                        ins_clause.part_title = after.part_title
                        ins_clause.section_title = after.section_title
        new_clauses.extend(insertions)

        # Transitional provisions (managed via temporal_policy resolver, not indexed as substantive clauses)
        # new_clauses.extend(_generate_transitional_clauses(amendment))

    return new_clauses
