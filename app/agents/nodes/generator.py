import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._utils import format_chunks, format_history, get_llm, load_prompt
from app.agents.graph.state import AgentState
from app.guardrails.output_guard import check as output_guard_check
from app.memory import session as session_mem
from app.observability import splunk


def _split_prefix(prefix: str) -> tuple[str, str, str]:
    parts = prefix.split(":", 2)
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else ("", "", "")


def generator_node(state: AgentState) -> dict:
    t0            = time.time()
    trajectory_id = state.get("trajectory_id", "")
    sid           = state.get("session_id", "")
    iteration     = state.get("iteration_count", 0)

    splunk.node_step(node="generator", phase="enter", trajectory_id=trajectory_id,
                     session_id=sid, iteration_count=iteration)

    prompt  = load_prompt("generator")
    chunks  = state.get("retrieved_chunks") or []
    history = session_mem.get_history(sid, *_split_prefix(sid)) if sid else []

    system_text = prompt["system"].format(
        retrieved_chunks=format_chunks(chunks),
        session_history=format_history(history),
    )

    response = get_llm().invoke([
        SystemMessage(content=system_text),
        HumanMessage(content=state["question"]),
    ])
    answer = response.content.strip()

    # Output guardrail — appends a warning note rather than hard-blocking
    guard = output_guard_check(answer, chunks)
    if guard.blocked:
        answer += f"\n\n> **Quality note:** {guard.reason}"
        splunk.security_event(
            event_type="output_guard_block",
            guard_type="output",
            reason=guard.reason,
            trajectory_id=trajectory_id,
            session_id=sid,
        )

    sources = [
        {
            "page":    c.get("page"),
            "section": c.get("section", ""),
            "source":  c.get("source", ""),
            "score":   c.get("score", 0.0),
        }
        for c in chunks[:5]
    ]

    # Persist exchange — msg_id ties each message to this trajectory run
    # so a retried generator call never duplicates history entries
    if sid:
        session_mem.append_message(sid, "user",      state["question"], msg_id=f"{trajectory_id}:user")
        session_mem.append_message(sid, "assistant", answer,            msg_id=f"{trajectory_id}:asst")

    splunk.node_step(
        node="generator", phase="exit",
        trajectory_id=trajectory_id, session_id=sid,
        iteration_count=iteration,
        duration_ms=round((time.time() - t0) * 1000, 2),
        answer_length=len(answer),
        source_count=len(sources),
        guard_blocked=guard.blocked,
    )

    return {
        "answer":     answer,
        "sources":    sources,
        "confidence": state.get("confidence", 0.5),
    }
