import pytest

from app.config import SETTINGS
from app.pipeline import PipelineContext, answer_question
from app.reasoning.contradiction_detector import detect_contradictions
from app.reasoning.decision import DecisionType
from app.reasoning.refusal import RefusalReason


@pytest.fixture(scope="module")
def ctx():
    if not (SETTINGS.index_path / "matrix.npy").exists():
        pytest.skip("Index not built. Run `python scripts/ingest.py` first.")
    return PipelineContext.load()


def test_known_contradiction_is_detected_corpus_wide(ctx):
    """§4.3.2 requires reporting within 10 calendar days. §9.1.4 refers to
    "the 30 calendar days required under §4.3" -- a direct, unsignalled
    numeric conflict about the same reporting obligation in original policy."""
    findings = detect_contradictions(ctx.store.clauses, ctx.clauses_by_id)
    pairs = {tuple(sorted([f.clause_a.clause_id, f.clause_b.clause_id])) for f in findings}
    assert ("4.3.2", "9.1.4") in pairs


def test_no_false_positive_on_explicitly_extended_periods(ctx):
    """§3.2.1 (28 days) and §3.2.2 (extended to 90 days) are a deliberate,
    signalled extension of the same period -- not a contradiction -- and
    must never be flagged as one."""
    findings = detect_contradictions(ctx.store.clauses, ctx.clauses_by_id)
    pairs = {tuple(sorted([f.clause_a.clause_id, f.clause_b.clause_id])) for f in findings}
    assert ("3.2.1", "3.2.2") not in pairs
    assert ("5.2.1", "5.2.2") not in pairs


def test_contradiction_count_is_small_and_precise(ctx):
    """A guard against threshold drift: this detector is deliberately
    tuned for precision over recall on a corpus this size (see
    DECISIONS.md) -- it should not be flagging a large fraction of the
    manual's ~150 clauses against each other."""
    findings = detect_contradictions(ctx.store.clauses, ctx.clauses_by_id)
    pairs = {tuple(sorted([f.clause_a.clause_id, f.clause_b.clause_id])) for f in findings}
    assert len(pairs) <= 3


def test_pipeline_surfaces_missing_date_refusal_for_reporting_deadline_question(ctx):
    """Post-Day 2: Reporting questions without a change date trigger a MISSING_DATE_CONTEXT refusal
    because Amendment No. 2026-01 introduced date-dependent rules (10 days vs 14 days)."""
    result = answer_question(
        "How long do I have to report a change in my income?", ctx
    )
    assert result.decision.decision_type == DecisionType.REFUSE
    assert result.decision.refusal_reason == RefusalReason.MISSING_DATE_CONTEXT
    assert "reporting period depends on when the change of circumstances occurred" in result.generation.text


def test_conflict_response_shows_missing_date_refusal(ctx):
    """Post-Day 2: Questions asking about late reporting overpayment deadlines without a date
    refuse safely due to missing date context."""
    result = answer_question(
        "If I report a change in my circumstances late, how many days "
        "do I have before it counts as an overpayment?",
        ctx,
    )
    assert result.decision.decision_type == DecisionType.REFUSE
    assert result.decision.refusal_reason == RefusalReason.MISSING_DATE_CONTEXT
