"""Unit tests for Pydantic schemas — validation and defaults."""

import pytest
from pydantic import ValidationError
from app.models.schemas import QueryRequest, AgentRequest, AgentResponse, AgentSource


# ── QueryRequest ──────────────────────────────────────────────────────────────

def test_query_request_minimal():
    req = QueryRequest(question="What is working capital?")
    assert req.question == "What is working capital?"
    assert req.top_k == 5
    assert req.tenant_id is None
    assert req.user_id is None
    assert req.session_id is None

def test_query_request_strips_whitespace():
    req = QueryRequest(question="  What is EBITDA?  ")
    assert req.question == "What is EBITDA?"

def test_query_request_empty_question_fails():
    with pytest.raises(ValidationError):
        QueryRequest(question="   ")

def test_query_request_too_long_fails():
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 501)

def test_query_request_top_k_min_fails():
    with pytest.raises(ValidationError):
        QueryRequest(question="test?", top_k=0)

def test_query_request_top_k_max_fails():
    with pytest.raises(ValidationError):
        QueryRequest(question="test?", top_k=11)

def test_query_request_top_k_valid_range():
    for k in [1, 5, 10]:
        req = QueryRequest(question="test?", top_k=k)
        assert req.top_k == k

def test_query_request_optional_ids():
    req = QueryRequest(question="test?", tenant_id="acme", user_id="u1", session_id="s1")
    assert req.tenant_id == "acme"


# ── AgentRequest ──────────────────────────────────────────────────────────────

def test_agent_request_minimal():
    req = AgentRequest(question="What is capital employed?")
    assert req.question == "What is capital employed?"
    assert req.session_id is None

def test_agent_request_with_session():
    req = AgentRequest(question="test?", session_id="sess-abc")
    assert req.session_id == "sess-abc"

def test_agent_request_empty_fails():
    with pytest.raises(ValidationError):
        AgentRequest(question="")

def test_agent_request_too_long_fails():
    with pytest.raises(ValidationError):
        AgentRequest(question="x" * 501)

def test_agent_request_strips_whitespace():
    req = AgentRequest(question="  What is WACC?  ")
    assert req.question == "What is WACC?"


# ── AgentResponse ─────────────────────────────────────────────────────────────

def test_agent_response_builds():
    resp = AgentResponse(
        answer="Working capital is...",
        sources=[AgentSource(page=63, section="WC", source="text", score=0.9)],
        confidence=0.85,
        plan=["query 1", "query 2"],
        iteration_count=1,
        session_id="sess-1",
        trajectory_id="traj-1",
        cache_hit=False,
        cache_type=None,
        similarity=None,
        idempotency_replay=False,
        created_at="2026-06-03T00:00:00Z",
        duration_ms=1234.5,
    )
    assert resp.confidence == 0.85
    assert len(resp.sources) == 1
    assert resp.sources[0].page == 63
