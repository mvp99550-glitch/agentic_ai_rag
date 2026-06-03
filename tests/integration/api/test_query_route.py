"""Integration tests for POST /api/v1/query — mocks Qdrant, Redis, PG."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


FAKE_CHUNKS = [
    {
        "text": "Working capital is defined as current assets minus current liabilities.",
        "page": 63,
        "section": "Working Capital",
        "source": "text",
        "score": 1.0,
        "collection": "finance_content",
    },
    {
        "text": "Capital employed is the sum of fixed assets and working capital.",
        "page": 65,
        "section": "Capital Employed",
        "source": "text",
        "score": 0.82,
        "collection": "finance_content",
    },
]

HEADERS = {
    "X-Tenant-Id":  "test_tenant",
    "X-User-Id":    "test_user",
    "X-Session-Id": "test_session",
}


@pytest.fixture
def client():
    with (
        patch("app.retrieval.search._embed"),
        patch("app.retrieval.search._sparse"),
        patch("app.retrieval.search._qdrant"),
        patch("app.memory.store.setup_tables"),
        patch("app.memory.store.ping", return_value=True),
        patch("app.memory.session.ping", return_value=True),
        patch("app.retrieval.reranker.warmup"),
        patch("app.observability.splunk.start_metrics"),
    ):
        from app.main_api import app
        yield TestClient(app, raise_server_exceptions=False)


def test_query_blocked_returns_400(client):
    with patch("app.api.routes.query.guard_check") as mock_guard:
        mock_guard.return_value = MagicMock(blocked=True, reason="SQL injection detected.")
        with patch("app.observability.splunk.security_event"):
            resp = client.post(
                "/api/v1/query",
                json={"question": "SELECT * FROM users"},
                headers=HEADERS,
            )
    assert resp.status_code == 400
    assert "SQL injection" in resp.json()["detail"]


def test_query_success_returns_chunks(client):
    with (
        patch("app.api.routes.query.guard_check") as mock_guard,
        patch("app.api.routes.query.hybrid_search", return_value=FAKE_CHUNKS),
        patch("app.api.routes.query.session_mem.append_message"),
        patch("app.api.routes.query.pg_store.log_query"),
        patch("app.observability.splunk.retrieval_query"),
    ):
        mock_guard.return_value = MagicMock(blocked=False, reason="")
        resp = client.post(
            "/api/v1/query",
            json={"question": "What is working capital?"},
            headers=HEADERS,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["question"] == "What is working capital?"
    assert len(data["chunks"]) > 0
    assert data["chunks"][0]["score"] >= 0.4   # threshold filter
    assert data["answer"] is None
    assert data["llm_ready"] is False


def test_query_score_threshold_filters_low_chunks(client):
    low_score_chunks = [
        {**FAKE_CHUNKS[0], "score": 0.25},
        {**FAKE_CHUNKS[1], "score": 0.50},
    ]
    with (
        patch("app.api.routes.query.guard_check") as mock_guard,
        patch("app.api.routes.query.hybrid_search", return_value=low_score_chunks),
        patch("app.api.routes.query.session_mem.append_message"),
        patch("app.api.routes.query.pg_store.log_query"),
        patch("app.observability.splunk.retrieval_query"),
    ):
        mock_guard.return_value = MagicMock(blocked=False, reason="")
        resp = client.post(
            "/api/v1/query",
            json={"question": "What is working capital?"},
            headers=HEADERS,
        )

    assert resp.status_code == 200
    chunks = resp.json()["chunks"]
    # Only score=0.50 chunk should pass threshold 0.4
    assert all(c["score"] >= 0.4 for c in chunks)


def test_query_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
