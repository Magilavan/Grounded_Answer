"""
Top-level orchestration: question in, GenerationResult out.

This function is intentionally thin -- it is the
`candidates = retrieve(...); evidence = verify_evidence(...); decision =
decide(...); response = generate_response(...)` shape the brief asks for
in section 34, so that any one stage can be swapped (a different
reranker, a different LLM, a different relevance threshold) without
touching the others. The CLI and the evaluation harness both call this
same function, so they can never observe different behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings, SETTINGS
from app.generation.generator import GenerationResult, generate_response
from app.reasoning.decision import Decision, decide
from app.retrieval.bm25_search import BM25Index, bm25_search
from app.retrieval.reranker import Reranker, build_idf_lookup, get_default_reranker
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.vector_search import vector_search
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger("grounded_answer.pipeline")


class EmptyQuestionError(ValueError):
    pass


@dataclass
class PipelineContext:
    """Everything the pipeline needs, built once at startup and reused
    across every question in a CLI session or evaluation run."""

    store: VectorStore
    bm25_index: BM25Index
    reranker: Reranker
    settings: Settings

    @classmethod
    def load(cls, settings: Settings | None = None) -> "PipelineContext":
        settings = settings or SETTINGS
        store = VectorStore.load(settings.index_path)
        bm25_index = BM25Index()
        bm25_index.build(store.clauses)
        idf = build_idf_lookup(store.clauses)
        return cls(
            store=store,
            bm25_index=bm25_index,
            reranker=get_default_reranker(idf=idf),
            settings=settings,
        )

    @property
    def clauses_by_id(self) -> dict:
        return {c.clause_id: c for c in self.store.clauses}


from app.reasoning.date_extraction import DateContext, extract_date_context
from app.reasoning.refusal import RefusalReason
from app.reasoning.temporal_policy import TemporalResolution, resolve_temporal_policy


@dataclass
class PipelineResult:
    generation: GenerationResult
    decision: Decision
    vector_hit_count: int
    bm25_hit_count: int
    fused_hit_count: int
    date_context: DateContext | None = None
    temporal_resolution: TemporalResolution | None = None


def answer_question(question: str, ctx: PipelineContext) -> PipelineResult:
    question = (question or "").strip()
    if not question:
        raise EmptyQuestionError("Question must not be empty.")

    settings = ctx.settings

    # 1. Date Context Extraction
    date_ctx = extract_date_context(question)
    if date_ctx.has_any_date:
        logger.info(
            "Extracted Date Context: claim=%s, change=%s, determination=%s, period=%s..%s",
            date_ctx.claim_date,
            date_ctx.change_date,
            date_ctx.determination_date,
            date_ctx.period_start,
            date_ctx.period_end,
        )

    # 2. Hybrid Retrieval
    search_query = date_ctx.cleaned_question or question
    vector_hits = vector_search(ctx.store, search_query, top_k=settings.retrieval_top_k)
    bm25_hits = bm25_search(ctx.bm25_index, search_query, top_k=settings.retrieval_top_k)
    logger.info("Vector candidates: %d", len(vector_hits))
    logger.info("BM25 candidates: %d", len(bm25_hits))

    fused = reciprocal_rank_fusion([vector_hits, bm25_hits], k=settings.rrf_k)
    logger.info("RRF candidates: %d", len(fused))

    top_fused = fused[: settings.retrieval_top_k]
    reranked = ctx.reranker.rerank(search_query, top_fused)
    reranked = reranked[: settings.rerank_top_k]
    logger.info("Reranked candidates: %d", len(reranked))

    # 3. Temporal Policy Resolution
    import copy
    raw_hits = []
    for hit, score in reranked:
        h = copy.copy(hit)
        h.score = score
        raw_hits.append(h)
    temp_res = resolve_temporal_policy(raw_hits, date_ctx, ctx.store.clauses)

    if temp_res.applicable_hits:
        filtered_reranked = [(hit, hit.score) for hit in temp_res.applicable_hits]
    else:
        filtered_reranked = reranked

    # 4. Handle Missing Date Context Refusal
    if temp_res.date_required_but_missing and not date_ctx.has_any_date:
        # Check if the highest-scoring clause is an amended topic
        if filtered_reranked and filtered_reranked[0][1] >= settings.min_relevance_score:
            top_clause = filtered_reranked[0][0].clause
            amended_ids = {"4.3.2", "9.1.4", "6.4.1", "6.6.1", "10.5.2", "10.5.3A"}
            if top_clause.clause_id in amended_ids or top_clause.is_amended_version:
                from app.reasoning.decision import DecisionType
                from app.reasoning.evidence import EvidenceAssessment
                detail_msg = temp_res.notes[0] if temp_res.notes else None
                missing_decision = Decision(
                    decision_type=DecisionType.REFUSE,
                    refusal_reason=RefusalReason.MISSING_DATE_CONTEXT,
                    evidence=EvidenceAssessment(
                        sufficient=False, items=[], supporting=[], reason=RefusalReason.MISSING_DATE_CONTEXT, detail=detail_msg
                    ),
                    contradictions=[],
                )
                generation = generate_response(
                    question=question,
                    decision=missing_decision,
                    all_clauses_by_id=ctx.clauses_by_id,
                    settings=settings,
                )
                return PipelineResult(
                    generation=generation,
                    decision=missing_decision,
                    vector_hit_count=len(vector_hits),
                    bm25_hit_count=len(bm25_hits),
                    fused_hit_count=len(fused),
                    date_context=date_ctx,
                    temporal_resolution=temp_res,
                )

    excluded_clause_ids = {h.clause.clause_id for h in temp_res.excluded_hits}

    # 5. Decision Logic
    decision = decide(
        question=question,
        reranked=filtered_reranked,
        clauses_by_id=ctx.clauses_by_id,
        min_relevance_score=settings.min_relevance_score,
        min_support_score=settings.min_support_score,
        excluded_clause_ids=excluded_clause_ids,
    )
    logger.info("Decision: %s", decision.decision_type.value)
    if decision.refusal_reason:
        logger.info("Reason: %s", decision.refusal_reason.value)

    # 6. Response Generation
    generation = generate_response(
        question=question,
        decision=decision,
        all_clauses_by_id=ctx.clauses_by_id,
        settings=settings,
    )

    # Append transitional citations if any were flagged by temporal resolution
    if temp_res.transitional_citations and generation.citations:
        for t_cit in temp_res.transitional_citations:
            if t_cit not in generation.citations:
                generation.citations.append(t_cit)

    if generation.citations:
        logger.info("Citations: %s", ", ".join(generation.citations))

    return PipelineResult(
        generation=generation,
        decision=decision,
        vector_hit_count=len(vector_hits),
        bm25_hit_count=len(bm25_hits),
        fused_hit_count=len(fused),
        date_context=date_ctx,
        temporal_resolution=temp_res,
    )
