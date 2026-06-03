"""Unit tests for Splunk HEC client — queue behavior, event structure."""

import queue
import time
from unittest.mock import patch

import pytest
import app.observability.splunk as splunk_mod


@pytest.fixture(autouse=True)
def reset_queue():
    """Drain the queue before each test so tests don't interfere."""
    while not splunk_mod._QUEUE.empty():
        try:
            splunk_mod._QUEUE.get_nowait()
        except queue.Empty:
            break
    yield


# ── _send ─────────────────────────────────────────────────────────────────────

def test_send_enqueues_correct_payload():
    with patch.object(splunk_mod, "_ensure_started"):
        splunk_mod._send("rag_pipeline", "rag:query", {"event_type": "test"})

    payload = splunk_mod._QUEUE.get_nowait()
    assert payload["index"] == "rag_pipeline"
    assert payload["sourcetype"] == "rag:query"
    assert payload["event"]["event_type"] == "test"
    assert "time" in payload
    assert isinstance(payload["time"], float)


def test_send_timestamp_is_recent():
    with patch.object(splunk_mod, "_ensure_started"):
        before = time.time()
        splunk_mod._send("idx", "src", {"k": "v"})
        after = time.time()

    payload = splunk_mod._QUEUE.get_nowait()
    assert before <= payload["time"] <= after


def test_send_never_raises_on_full_queue():
    with patch.object(splunk_mod, "_ensure_started"):
        with patch.object(splunk_mod._QUEUE, "put_nowait", side_effect=queue.Full):
            splunk_mod._send("idx", "src", {"k": "v"})  # must not raise


def test_send_never_raises_on_any_exception():
    with patch.object(splunk_mod, "_ensure_started"):
        with patch.object(splunk_mod._QUEUE, "put_nowait", side_effect=RuntimeError("boom")):
            splunk_mod._send("idx", "src", {"k": "v"})  # must not raise


# ── Helper functions ──────────────────────────────────────────────────────────

def test_rag_query_sends_to_rag_pipeline():
    with patch.object(splunk_mod, "_ensure_started"):
        splunk_mod.rag_query(
            tenant_id="acme", user_id="u1", session_id="s1",
            trajectory_id="t1", question_length=30, answer_length=200,
            confidence=0.85, cache_hit=False, cache_type=None,
            iteration_count=1, duration_ms=1234.0, source_count=3,
        )
    payload = splunk_mod._QUEUE.get_nowait()
    assert payload["index"] == "rag_pipeline"
    assert payload["event"]["event_type"] == "rag_query"
    assert payload["event"]["confidence"] == 0.85


def test_node_step_sends_to_agent_trajectory():
    with patch.object(splunk_mod, "_ensure_started"):
        splunk_mod.node_step(
            node="planner", phase="enter",
            trajectory_id="t1", session_id="s1",
        )
    payload = splunk_mod._QUEUE.get_nowait()
    assert payload["index"] == "agent_trajectory"
    assert payload["event"]["node"] == "planner"
    assert payload["event"]["phase"] == "enter"


def test_node_step_exit_includes_duration():
    with patch.object(splunk_mod, "_ensure_started"):
        splunk_mod.node_step(
            node="generator", phase="exit",
            trajectory_id="t1", session_id="s1",
            duration_ms=456.7,
        )
    payload = splunk_mod._QUEUE.get_nowait()
    assert payload["event"]["duration_ms"] == 456.7


def test_security_event_sends_to_security_events():
    with patch.object(splunk_mod, "_ensure_started"):
        splunk_mod.security_event(
            event_type="input_guard_block",
            guard_type="input",
            reason="SQL injection",
            tenant_id="acme",
            question_length=50,
        )
    payload = splunk_mod._QUEUE.get_nowait()
    assert payload["index"] == "security_events"
    assert payload["event"]["event_type"] == "input_guard_block"
    assert payload["event"]["reason"] == "SQL injection"
