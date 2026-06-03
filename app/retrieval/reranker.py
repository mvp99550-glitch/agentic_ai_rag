"""
Cross-encoder reranker — precision pass after hybrid retrieval.

Model: BAAI/bge-reranker-base (same family as bge-base-en-v1.5 bi-encoder).

Usage:
    from app.retrieval.reranker import rerank

    chunks = hybrid_search(query, "finance_content", top_k=15)
    chunks = rerank(query, chunks, top_k=5)
"""

from sentence_transformers import CrossEncoder

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    return _model


def warmup() -> None:
    """Load the cross-encoder into memory at startup so the first real request isn't slow."""
    _get_model()


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Score each (query, chunk_text) pair with the cross-encoder and return
    the top_k chunks sorted by reranker score descending.

    Each returned chunk gets an extra 'rerank_score' key so downstream
    nodes and the UI can show it alongside the original retrieval score.

    Fallback: if the cross-encoder fails for any reason, returns the top_k
    chunks sorted by their original hybrid retrieval score so the pipeline
    never stalls.
    """
    if not chunks:
        return chunks

    try:
        model  = _get_model()
        pairs  = [(query, c.get("text", "")) for c in chunks]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: float(x[0]),
            reverse=True,
        )
        return [
            {**chunk, "rerank_score": round(float(score), 4)}
            for score, chunk in ranked[:top_k]
        ]

    except Exception:
        # Degrade gracefully: skip reranking, sort by retrieval score
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
        return [
            {**c, "rerank_score": c.get("score", 0.0)}
            for c in sorted_chunks[:top_k]
        ]
