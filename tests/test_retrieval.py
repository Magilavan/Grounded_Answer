import pytest

from app.config import SETTINGS
from app.pipeline import PipelineContext
from app.retrieval.bm25_search import BM25Index, bm25_search
from app.retrieval.reranker import build_idf_lookup, get_default_reranker
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.vector_search import vector_search


@pytest.fixture(scope="module")
def ctx():
    if not (SETTINGS.index_path / "matrix.npy").exists():
        pytest.skip("Index not built. Run `python scripts/ingest.py` first.")
    return PipelineContext.load()


def test_vector_search_finds_exact_topic(ctx):
    hits = vector_search(ctx.store, "recipient obligation to report changes within days", top_k=8)
    ids = [h.clause.clause_id for h in hits]
    assert "4.3.2" in ids


def test_vector_search_handles_paraphrase(ctx):
    """Different wording than the clause itself should still retrieve it --
    this is the whole point of semantic over pure keyword search."""
    hits = vector_search(ctx.store, "what should I do if my earnings change", top_k=8)
    ids = [h.clause.clause_id for h in hits]
    assert "4.3.2" in ids


def test_bm25_finds_exact_terminology(ctx):
    index = BM25Index()
    index.build(ctx.store.clauses)
    hits = bm25_search(index, "calendar days overpayment", top_k=5)
    assert len(hits) > 0


def test_rrf_combines_and_orders_by_fused_score():
    from app.retrieval.vector_search import SearchHit
    from app.ingestion.chunker import Clause

    def make_clause(cid):
        return Clause(
            clause_id=cid, citation=f"§{cid}", part_number="1", part_title="T",
            section_number="1.1", section_title="S", text="text", order=1,
            line_start=1, line_end=1,
        )

    a, b, c = make_clause("1.1.1"), make_clause("1.1.2"), make_clause("1.1.3")
    list1 = [SearchHit(a, 0.9, "vector"), SearchHit(b, 0.5, "vector")]
    list2 = [SearchHit(b, 0.8, "bm25"), SearchHit(c, 0.4, "bm25")]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    ids = [h.clause.clause_id for h in fused]
    # b appears in both lists so should outrank a and c, which each
    # appear in only one list.
    assert ids[0] == "1.1.2"
    assert set(ids) == {"1.1.1", "1.1.2", "1.1.3"}


def test_reranker_prefers_distinctive_term_match(ctx):
    """Regression test for a real bug found during development: unweighted
    token overlap preferred a clause sharing only generic words ("needs
    figure", "household") over the clause that actually mentions the
    distinctive term the question turns on ("student")."""
    idf = build_idf_lookup(ctx.store.clauses)
    reranker = get_default_reranker(idf=idf)
    question = "How is the needs figure calculated for a household with a full-time student?"

    vh = vector_search(ctx.store, question, top_k=12)
    bh = bm25_search(BM25Index(), question, top_k=0)  # empty on purpose
    fused = reciprocal_rank_fusion([vh, bh], k=60)
    reranked = reranker.rerank(question, fused[:12])

    top_id = reranked[0][0].clause.clause_id
    assert top_id == "7.1.3"


def test_out_of_scope_question_scores_low(ctx):
    idf = build_idf_lookup(ctx.store.clauses)
    reranker = get_default_reranker(idf=idf)
    question = "What is the capital of France?"

    vh = vector_search(ctx.store, question, top_k=12)
    reranked = reranker.rerank(question, vh)
    if reranked:
        assert reranked[0][1] < SETTINGS.min_relevance_score
