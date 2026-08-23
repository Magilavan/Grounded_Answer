import pytest

from app.citations.validator import extract_citations, validate_citations
from app.config import SETTINGS
from app.pipeline import PipelineContext


@pytest.fixture(scope="module")
def ctx():
    if not (SETTINGS.index_path / "matrix.npy").exists():
        pytest.skip("Index not built. Run `python scripts/ingest.py` first.")
    return PipelineContext.load()


def test_extract_citations_finds_all_refs():
    text = "See §4.3.2 and also §9.1.4 for details."
    assert extract_citations(text) == ["4.3.2", "9.1.4"]


def test_valid_citation_from_supporting_evidence_passes(ctx):
    clause = ctx.clauses_by_id["4.3.2"]
    text = f"You must report within 10 days. Source: {clause.citation}"
    result = validate_citations(text, supporting_clauses=[clause], all_clauses_by_id=ctx.clauses_by_id)
    assert result.valid
    assert result.invalid_citations == []
    assert result.unsupported_citations == []


def test_hallucinated_clause_number_is_rejected(ctx):
    """A clause number that doesn't exist anywhere in the manual must
    never be accepted, regardless of how plausible the surrounding answer
    text sounds."""
    text = "You must report within 10 days. Source: §99.9.9"
    result = validate_citations(text, supporting_clauses=[], all_clauses_by_id=ctx.clauses_by_id)
    assert not result.valid
    assert "99.9.9" in result.invalid_citations


def test_real_clause_not_in_evidence_is_rejected(ctx):
    """A citation to a clause that genuinely exists in the manual, but was
    not part of the evidence actually used for this answer, must still be
    rejected -- existing in the manual is not sufficient grounding."""
    real_but_unused = ctx.clauses_by_id["6.1.1"]
    used_clause = ctx.clauses_by_id["4.3.2"]
    text = f"Some claim. Source: {real_but_unused.citation}"
    result = validate_citations(
        text, supporting_clauses=[used_clause], all_clauses_by_id=ctx.clauses_by_id
    )
    assert not result.valid
    assert "6.1.1" in result.unsupported_citations


def test_no_citations_is_trivially_valid(ctx):
    result = validate_citations("No citations here.", [], ctx.clauses_by_id)
    assert result.valid
    assert result.checked_citations == []
