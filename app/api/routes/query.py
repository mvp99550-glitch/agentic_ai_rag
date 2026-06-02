"""POST /api/v1/query — retrieval endpoint with citations and guardrails."""

import os
from fastapi import APIRouter, Depends, HTTPException, Request
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from app.models.schemas import QueryRequest, QueryResponse, RetrievedChunk, Citation
from app.api.middleware.context import RequestContext, get_context
from app.memory import session as session_mem
from app.memory import store as pg_store
from app.guardrails.input_guard import check as guard_check

router = APIRouter()

# ── Singletons ────────────────────────────────────────────────────────────────
_embed = SentenceTransformer("all-MiniLM-L6-v2")
_qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60,
)


def _search(query_vec: list[float], collection: str, top_k: int) -> list[dict]:
    return _qdrant.query_points(
        collection_name=collection,
        query=query_vec,
        limit=top_k,
    ).points


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, ctx: RequestContext = Depends(get_context)):

    # ── Guardrail check ───────────────────────────────────────────────────────
    # Blocked queries stop here — Qdrant, Redis, and PostgreSQL are never touched.
    result = guard_check(req.question)
    if result.blocked:
        raise HTTPException(status_code=400, detail=result.reason)

    # ── Embed ─────────────────────────────────────────────────────────────────
    query_vec = _embed.encode(req.question).tolist()

    # ── Retrieve ──────────────────────────────────────────────────────────────
    raw_content   = _search(query_vec, "finance_content",   req.top_k)
    raw_structure = _search(query_vec, "finance_structure", 2)

    # Deduplicate and assign citation numbers
    seen:   set[str]           = set()
    chunks: list[RetrievedChunk] = []
    citations: list[Citation]  = []
    num = 1

    for r in raw_content + raw_structure:
        text = r.payload.get("text", "")
        if text in seen:
            continue
        seen.add(text)

        collection = "finance_content" if r in raw_content else "finance_structure"
        chunks.append(RetrievedChunk(
            citation_number=num,
            text=text,
            score=round(r.score, 4),
            page=r.payload.get("page"),
            section=r.payload.get("section", ""),
            source=r.payload.get("source", ""),
            collection=collection,
        ))
        citations.append(Citation(
            number=num,
            page=r.payload.get("page"),
            section=r.payload.get("section", ""),
            source=r.payload.get("source", ""),
            collection=collection,
        ))
        num += 1

    # ── Short-term memory (Redis) ─────────────────────────────────────────────
    session_mem.append_message(ctx.redis_prefix, "user", req.question)

    # ── Long-term log (PostgreSQL) ────────────────────────────────────────────
    citation_summary = ", ".join(
        f"[{c.number}] p{c.page} {c.section}" for c in citations[:3]
    )
    pg_store.log_query(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        session_id=ctx.session_id,
        query=req.question,
        retrieved=citation_summary,
    )

    return QueryResponse(
        question=req.question,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        session_id=ctx.session_id,
        chunks=chunks,
        citations=citations,
        answer=None,
        llm_ready=False,
    )
