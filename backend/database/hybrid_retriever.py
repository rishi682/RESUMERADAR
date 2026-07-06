from rank_bm25 import BM25Okapi

from backend.database.vector_store import query as semantic_query


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into simple whitespace tokens for BM25."""
    return text.lower().split()


def _bm25_rank(query_text: str, corpus: list[str], doc_ids: list[str]) -> list[str]:
    """Return doc_ids ranked by BM25 score, descending."""
    tokenized_corpus = [_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query_text))
    ranked = sorted(zip(doc_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [doc_id for doc_id, _ in ranked]


def _semantic_rank(query_text: str, n_results: int) -> list[str]:
    """Return doc_ids ranked by semantic similarity via ChromaDB, ascending distance = best first."""
    results = semantic_query(query_text, n_results=n_results)
    return results["ids"][0] if results["ids"] else []


def _reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def hybrid_search(
    query_text: str, corpus: list[str], doc_ids: list[str], n_results: int = 5
) -> list[tuple[str, float]]:
    """Run BM25 and semantic search, then fuse rankings with RRF. corpus/doc_ids must be parallel lists of all candidate documents."""
    bm25_ranking = _bm25_rank(query_text, corpus, doc_ids)
    semantic_ranking = _semantic_rank(query_text, n_results=len(doc_ids))
    fused = _reciprocal_rank_fusion([bm25_ranking, semantic_ranking])
    return fused[:n_results]