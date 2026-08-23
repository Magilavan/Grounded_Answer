from pathlib import Path

from app.ingestion.chunker import parse_clauses
from app.ingestion.loader import ManualNotFoundError, load_manual_text

MANUAL_PATH = Path(__file__).resolve().parent.parent / "data" / "policy-manual.md"


def test_manual_loads():
    text = load_manual_text(MANUAL_PATH)
    assert len(text) > 1000


def test_missing_manual_raises_clear_error(tmp_path):
    missing = tmp_path / "does-not-exist.md"
    try:
        load_manual_text(missing)
        assert False, "expected ManualNotFoundError"
    except ManualNotFoundError as exc:
        assert "not found" in str(exc)


def test_clauses_parsed_with_no_duplicates():
    text = load_manual_text(MANUAL_PATH)
    clauses = parse_clauses(text)
    assert len(clauses) > 100
    ids = [c.clause_id for c in clauses]
    assert len(ids) == len(set(ids)), "clause ids must be unique"


def test_every_clause_has_citation_and_metadata():
    text = load_manual_text(MANUAL_PATH)
    clauses = parse_clauses(text)
    for c in clauses:
        assert c.citation == f"§{c.clause_id}"
        assert c.text.strip()
        meta = c.to_metadata()
        assert meta["clause"] == c.citation
        assert meta["section"]
        assert meta["part"]


def test_defined_term_clauses_keep_their_term():
    """§1.4.x clauses use the "**1.4.1 Applicant** — ..." style heading;
    the defined term must survive into the clause text, not get dropped."""
    text = load_manual_text(MANUAL_PATH)
    clauses = parse_clauses(text)
    by_id = {c.clause_id: c for c in clauses}
    applicant = by_id.get("1.4.1")
    assert applicant is not None
    assert "applicant" in applicant.text.lower()


def test_known_clauses_present():
    text = load_manual_text(MANUAL_PATH)
    clauses = parse_clauses(text)
    ids = {c.clause_id for c in clauses}
    for expected in ["4.3.2", "9.1.4", "7.1.3", "3.2.3", "5.2.3"]:
        assert expected in ids, f"expected clause §{expected} to be parsed"
