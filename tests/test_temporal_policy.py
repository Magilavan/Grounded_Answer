"""
Test suite for Day-2 Temporal Policy Resolution.

Verifies:
  - Reporting period pre-amendment (10 days) vs post-amendment (14 days)
  - Transitional provision §5.2 handling (change of circumstances date)
  - Earnings disregard pre-amendment ($120) vs post-amendment ($175) under §5.1
  - Income threshold pre-amendment vs post-amendment table replacement
  - Sanction rate reduction (20% -> 15%) and new exception (§10.5.3A)
  - Claim spanning 1 March 2026 boundary (apportionment per §5.3 / §7.4.3)
  - Refusal on missing date context for temporal-dependent rules
  - Same question with different dates producing distinct answers
"""

from __future__ import annotations

import pytest

from app.pipeline import PipelineContext, answer_question
from app.reasoning.decision import DecisionType
from app.reasoning.refusal import RefusalReason


@pytest.fixture(scope="module")
def pipeline_ctx():
    return PipelineContext.load()


def test_reporting_period_pre_amendment(pipeline_ctx):
    """A change of circumstances on 20 February 2026 requires reporting within 10 calendar days."""
    q = "A recipient had a change of circumstances on 20 February 2026. How many days do they have to report it?"
    res = answer_question(q, pipeline_ctx)

    assert res.decision.decision_type == DecisionType.ANSWER
    assert "10" in res.generation.text
    assert any("4.3.2" in c for c in res.generation.citations)


def test_reporting_period_post_amendment(pipeline_ctx):
    """A change of circumstances on 10 April 2026 requires reporting within 14 calendar days."""
    q = "A recipient had a change of circumstances on 10 April 2026. How many days do they have to report it?"
    res = answer_question(q, pipeline_ctx)

    assert res.decision.decision_type == DecisionType.ANSWER
    assert "14" in res.generation.text
    assert any("4.3.2" in c for c in res.generation.citations)


def test_reporting_transitional_provision_cited(pipeline_ctx):
    """Post-amendment reporting queries include or reference §5.2 transitional provision."""
    q = "What is the reporting period for a change of circumstances occurring on 15 March 2026?"
    res = answer_question(q, pipeline_ctx)

    assert res.decision.decision_type == DecisionType.ANSWER
    assert any("4.3.2" in c for c in res.generation.citations)
    # Check that transitional citation or note is attached
    has_transitional = any("5.2" in c for c in res.generation.citations) or (
        res.temporal_resolution and any("5.2" in n for n in res.temporal_resolution.notes)
    )
    assert has_transitional


def test_earnings_disregard_pre_amendment(pipeline_ctx):
    """A determination made on 15 February 2026 applies the $120 earnings disregard."""
    q = "What is the monthly earnings disregard for a determination made on 15 February 2026?"
    res = answer_question(q, pipeline_ctx)

    assert res.decision.decision_type == DecisionType.ANSWER
    assert "120" in res.generation.text
    assert any("6.4.1" in c for c in res.generation.citations)


def test_earnings_disregard_post_amendment(pipeline_ctx):
    """A determination made on 15 March 2026 applies the $175 earnings disregard."""
    q = "What is the monthly earnings disregard for a determination made on 15 March 2026?"
    res = answer_question(q, pipeline_ctx)

    assert res.decision.decision_type == DecisionType.ANSWER
    assert "175" in res.generation.text
    assert any("6.4.1" in c for c in res.generation.citations)


def test_income_threshold_pre_amendment(pipeline_ctx):
    """A determination made on 10 February 2026 uses pre-amendment thresholds ($1,180 for single)."""
    q = "What is the income threshold for a single person for a determination made on 10 February 2026?"
    res = answer_question(q, pipeline_ctx)

    assert res.decision.decision_type == DecisionType.ANSWER
    assert "1,180" in res.generation.text or "1180" in res.generation.text
    assert any("6.6.1" in c for c in res.generation.citations)


def test_income_threshold_post_amendment(pipeline_ctx):
    """A determination made on 10 March 2026 uses updated thresholds ($1,225 for single)."""
    q = "What is the income threshold for a single person for a determination made on 10 March 2026?"
    res = answer_question(q, pipeline_ctx)

    assert res.decision.decision_type == DecisionType.ANSWER
    assert "1,225" in res.generation.text or "1225" in res.generation.text
    assert any("6.6.1" in c for c in res.generation.citations)


def test_sanction_reduction_pre_vs_post(pipeline_ctx):
    """Sanction reduction is 20 per cent pre-March 2026 and 15 per cent post-March 2026."""
    q_pre = "What is the sanction reduction rate for a determination made on 1 February 2026?"
    res_pre = answer_question(q_pre, pipeline_ctx)
    assert res_pre.decision.decision_type == DecisionType.ANSWER
    assert "20" in res_pre.generation.text

    q_post = "What is the sanction reduction rate for a determination made on 10 March 2026?"
    res_post = answer_question(q_post, pipeline_ctx)
    assert res_post.decision.decision_type == DecisionType.ANSWER
    assert "15" in res_post.generation.text


def test_new_sanction_exception_post_amendment(pipeline_ctx):
    """Sanction exception under §10.5.3A applies post-amendment."""
    q = "Can a sanction be imposed for failure to report if the change of circumstances would have increased the award, for a determination on 5 March 2026?"
    res = answer_question(q, pipeline_ctx)

    assert res.decision.decision_type == DecisionType.ANSWER
    assert any("10.5.3A" in c or "10.5.3" in c for c in res.generation.citations)


def test_claim_spanning_boundary_apportionment(pipeline_ctx):
    """A claim period spanning 1 March 2026 triggers apportionment per §5.3 / §7.4.3."""
    q = "How is an award calculated for a claim period running from 15 February 2026 through 15 March 2026?"
    res = answer_question(q, pipeline_ctx)

    assert res.decision.decision_type == DecisionType.ANSWER
    assert (
        "apportion" in res.generation.text.lower()
        or "7.4.3" in res.generation.text
        or "5.3" in str(res.generation.citations)
    )


def test_missing_date_context_refusal(pipeline_ctx):
    """A question asking about the reporting period without specifying a date should refuse or request clarification."""
    q = "How many calendar days do I have to report a change of circumstances?"
    res = answer_question(q, pipeline_ctx)

    assert res.decision.decision_type == DecisionType.REFUSE
    assert res.decision.refusal_reason == RefusalReason.MISSING_DATE_CONTEXT


def test_same_question_different_date_produces_different_answers(pipeline_ctx):
    """Demonstrates that identical questions with different date contexts produce different answers."""
    q1 = "How many days does a recipient have to report a change of circumstances occurring on 10 February 2026?"
    q2 = "How many days does a recipient have to report a change of circumstances occurring on 10 April 2026?"

    res1 = answer_question(q1, pipeline_ctx)
    res2 = answer_question(q2, pipeline_ctx)

    assert res1.decision.decision_type == DecisionType.ANSWER
    assert res2.decision.decision_type == DecisionType.ANSWER
    assert "10" in res1.generation.text
    assert "14" in res2.generation.text
    assert res1.generation.text != res2.generation.text
