"""
Shared hybrid retrieval singletons — BAAI/bge-base-en-v1.5 (768d) + BM25.
Imported by both the FastAPI query route and the agent retriever node so the
embedding models are loaded once per process, not once per module.
"""

import os

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import FusionQuery, Fusion, Prefetch, SparseVector
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

_embed  = SentenceTransformer("BAAI/bge-base-en-v1.5")
_sparse = SparseTextEmbedding(model_name="Qdrant/bm25")
_qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _query_qdrant(collection: str, prefetch: list, query, limit: int):
    """Retried Qdrant network call — 3 attempts, 1→2→4→8s backoff."""
    return _qdrant.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=query,
        limit=limit,
    ).points


def hybrid_search(query: str, collection: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid dense+sparse search with RRF fusion.
    Returns list of dicts: {text, page, section, source, score, collection}.
    Qdrant call is retried up to 3 times on transient errors.
    """
    dense_vec  = _embed.encode(query).tolist()
    sparse_raw = list(_sparse.query_embed(query))[0]
    sparse_vec = SparseVector(
        indices=sparse_raw.indices.tolist(),
        values=sparse_raw.values.tolist(),
    )
    points = _query_qdrant(
        collection,
        prefetch=[
            Prefetch(query=dense_vec,  using="dense",  limit=top_k * 2),
            Prefetch(query=sparse_vec, using="sparse", limit=top_k * 2),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    )
    return [
        {
            "text":       r.payload.get("text", ""),
            "page":       r.payload.get("page"),
            "section":    r.payload.get("section", ""),
            "source":     r.payload.get("source", ""),
            "score":      round(r.score, 4),
            "collection": collection,
        }
        for r in points
    ]
