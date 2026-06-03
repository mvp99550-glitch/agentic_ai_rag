"""Unit tests for agent _utils — parse_json, format_chunks, format_history."""

import pytest
from app.agents._utils import parse_json, format_chunks, format_history


# ── parse_json ────────────────────────────────────────────────────────────────

def test_parse_valid_json():
    assert parse_json('{"key": "value"}') == {"key": "value"}

def test_parse_json_with_markdown_fence():
    text = '```json\n{"sub_tasks": ["task1"]}\n```'
    assert parse_json(text) == {"sub_tasks": ["task1"]}

def test_parse_json_fence_without_language():
    text = '```\n{"key": "value"}\n```'
    assert parse_json(text) == {"key": "value"}

def test_parse_invalid_json_returns_empty():
    assert parse_json("not json at all") == {}

def test_parse_empty_string_returns_empty():
    assert parse_json("") == {}

def test_parse_json_string_returns_empty():
    # LLM returns a bare JSON string — was the original bug
    assert parse_json('"just a string"') == {}

def test_parse_json_list_returns_empty():
    # LLM returns a list instead of object
    assert parse_json('["item1", "item2"]') == {}

def test_parse_json_number_returns_empty():
    assert parse_json("42") == {}

def test_parse_json_nested():
    text = '{"plan": ["a", "b"], "confidence": 0.9}'
    result = parse_json(text)
    assert result["confidence"] == 0.9
    assert len(result["plan"]) == 2

def test_parse_json_always_returns_dict():
    # Guarantee: result is always a dict, never raises
    for bad in ['"string"', '[1,2]', 'null', 'true', '123', '', '{bad}']:
        result = parse_json(bad)
        assert isinstance(result, dict)


# ── format_chunks ─────────────────────────────────────────────────────────────

def test_format_chunks_empty():
    assert format_chunks([]) == "No chunks retrieved."

def test_format_chunks_single():
    chunks = [{"page": 63, "section": "Working Capital", "text": "WC is..."}]
    result = format_chunks(chunks)
    assert "[1]" in result
    assert "Page 63" in result
    assert "Working Capital" in result
    assert "WC is..." in result

def test_format_chunks_multiple():
    chunks = [
        {"page": 63, "section": "A", "text": "text A"},
        {"page": 65, "section": "B", "text": "text B"},
    ]
    result = format_chunks(chunks)
    assert "[1]" in result
    assert "[2]" in result

def test_format_chunks_respects_max():
    chunks = [{"page": i, "section": "S", "text": f"t{i}"} for i in range(20)]
    result = format_chunks(chunks, max_chunks=3)
    assert "[3]" in result
    assert "[4]" not in result

def test_format_chunks_missing_fields():
    chunks = [{"text": "no page or section"}]
    result = format_chunks(chunks)
    assert "no page or section" in result


# ── format_history ────────────────────────────────────────────────────────────

def test_format_history_empty():
    assert format_history([]) == "No prior conversation."

def test_format_history_single_turn():
    history = [{"role": "user", "content": "What is EBITDA?"}]
    result = format_history(history)
    assert "User: What is EBITDA?" in result

def test_format_history_two_turns():
    history = [
        {"role": "user",      "content": "What is EBITDA?"},
        {"role": "assistant", "content": "EBITDA is..."},
    ]
    result = format_history(history)
    assert "User:" in result
    assert "Assistant:" in result

def test_format_history_last_n():
    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    result = format_history(history, last_n=2)
    assert "q8" in result
    assert "q9" in result
    assert "q0" not in result
