import pytest

from app.config import SETTINGS
from app.pipeline import PipelineContext, answer_question
from app.reasoning.decision import DecisionType
from app.reasoning.refusal import RefusalReason


@pytest.fixture(scope="module")
def ctx():
    if not (SETTINGS.index_path / "matrix.npy").exists():
        pytest.skip("Index not built. Run `python scripts/ingest.py` first.")
    return PipelineContext.load()


def test_out_of_scope_question_is_refused(ctx):
    result = answer_question("What is the capital of France?", ctx)
    assert result.decision.decision_type == DecisionType.REFUSE
    assert result.decision.refusal_reason == RefusalReason.INSUFFICIENT_EVIDENCE
    assert result.generation.citations == []


def test_apparent_gap_question_is_refused_not_answered(ctx):
    """The core failure mode this whole project targets: a passage that
    looks like it covers the question (§7.1.3 explicitly mentions
    full-time students) but, read closely, defers to a clause (§5.4) that
    is actually about something else entirely (care allowances). A naive
    similarity-based system would confidently answer from §7.1.3; this
    system must refuse instead."""
    result = answer_question(
        "How is the needs figure calculated for a household that includes "
        "a full-time student?",
        ctx,
    )
    assert result.decision.decision_type == DecisionType.REFUSE
    assert result.decision.refusal_reason == RefusalReason.AMBIGUOUS_POLICY
    assert result.generation.citations == []


def test_second_apparent_gap_bare_deferral_is_refused(ctx):
    """§3.2.3 / §5.2.3 both say full-time-education absence is "addressed
    separately" without naming any clause -- and no such clause exists."""
    result = answer_question(
        "Does a household member who is away at university full-time "
        "still count as a household member for the assistance calculation?",
        ctx,
    )
    assert result.decision.decision_type == DecisionType.REFUSE


def test_refusal_always_includes_escalation_guidance(ctx):
    result = answer_question("What is the capital of France?", ctx)
    assert "administrator" in result.generation.text.lower() or "supervisor" in result.generation.text.lower()


def test_empty_question_raises():
    from app.pipeline import EmptyQuestionError

    with pytest.raises(EmptyQuestionError):
        from app.pipeline import PipelineContext
        # Doesn't need a real index -- validated before retrieval runs.
        class DummyCtx:
            settings = SETTINGS
        answer_question("   ", DummyCtx())


def test_direct_answerable_question_is_answered_with_citation(ctx):
    result = answer_question(
        "What is the definition of an applicant under the program?", ctx
    )
    assert result.decision.decision_type == DecisionType.ANSWER
    assert result.generation.citations
    assert "1.4.1" in [c.lstrip("§") for c in result.generation.citations]
