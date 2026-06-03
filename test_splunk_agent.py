"""
Quick smoke test — runs the full agent pipeline and prints results.
Fires rag_pipeline (rag:query) + agent_trajectory events to Splunk.

Usage:
    uv run python test_splunk_agent.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from app.agents.graph.stateful import run

result = run(
    question="Explain the relationship between EBITDA and operating cash flow.",
    tenant_id="acme",
    user_id="user1",
    session_id="splunk-test-01",
)

print(f"Answer length : {len(result['answer'])} chars")
print(f"Confidence    : {result['confidence']:.2f}")
print(f"Iterations    : {result['iteration_count']}")
print(f"Duration      : {result['duration_ms']} ms")
print(f"Cache hit     : {result['cache_hit']}")
print(f"Trajectory ID : {result['trajectory_id']}")
print(f"\nAnswer preview:\n{result['answer'][:300]}...")
