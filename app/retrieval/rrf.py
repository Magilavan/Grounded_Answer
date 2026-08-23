"""
Reciprocal Rank Fusion (RRF) of the vector and BM25 result lists.

RRF combines two rankings using each item's *rank position* rather than its
raw score, which sidesteps the problem that TF-IDF cosine similarity and
BM25 scores live on different, incomparable scales. See DECISIONS.md for
why RRF was chosen over score-normalisation-and-averaging.

score(d) = sum over each ranking r that contains d of  1 / (k + rank_r(d))
"""

from __future__ import annotations

from app.retrieval.vector_search import SearchHit


def reciprocal_rank_fusion(
    result_lists: list[list[SearchHit]], k: int = 60
) -> list[SearchHit]:
    """Fuse multiple ranked lists of SearchHit into one, ordered list.

    Hits are keyed by clause_id: if the same clause appears in more than
    one input list, its RRF scores are summed and the highest-scoring
    retriever's SearchHit is kept as the representative record (its
    `.retriever` field is overwritten to "hybrid" to make that visible
    downstream).
    """
    fused_scores: dict[str, float] = {}
    representative: dict[str, SearchHit] = {}
    contributing_retrievers: dict[str, set[str]] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list, start=1):
            cid = hit.clause.clause_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank)
            contributing_retrievers.setdefault(cid, set()).add(hit.retriever)
            # Keep the hit with the higher original score as representative
            # so downstream logging can show which retriever "found" it
            # most strongly.
            if cid not in representative or hit.score > representative[cid].score:
                representative[cid] = hit

    fused_hits: list[SearchHit] = []
    for cid, score in fused_scores.items():
        base = representative[cid]
        retrievers = contributing_retrievers[cid]
        label = "hybrid" if len(retrievers) > 1 else next(iter(retrievers))
        fused_hits.append(SearchHit(clause=base.clause, score=score, retriever=label))

    fused_hits.sort(key=lambda h: h.score, reverse=True)
    return fused_hits
