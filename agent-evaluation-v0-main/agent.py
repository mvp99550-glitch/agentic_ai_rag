"""
Groq-based QA agent with tool use for multi-hop question answering.

This is the agent being evaluated. It starts deliberately naive (minimal system
prompt, simple tool loop) so that Phase 2's autoresearch loop has room to improve it.

The agent has two tools:
- search_paragraphs(query): keyword search over the provided context paragraphs
- read_paragraph(title): read all sentences of a specific paragraph by title

The agent loop: send question → if model uses tools, execute them and loop back →
when model responds with text (no tools), return that as the answer.

Phase 2 can edit this file and system_prompt.md to improve scores.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langfuse import get_client

load_dotenv()

langfuse = get_client()
MODEL    = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = Path("system_prompt.md").read_text().strip()

# ── Tool definitions ──────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "search_paragraphs",
        "description": "Search context paragraphs by keyword. Returns paragraphs whose title or sentences contain the query terms.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant paragraphs",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_paragraph",
        "description": "Read all sentences of a specific paragraph by its exact title.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The exact title of the paragraph to read",
                }
            },
            "required": ["title"],
        },
    },
]

llm = ChatGroq(
    model=MODEL,
    api_key=os.environ["GROQ_API_KEY"],
    max_tokens=8192,
    temperature=1,
).bind_tools(TOOLS)


# ── Tool executors ────────────────────────────────────────────────────────────

def execute_search_paragraphs(query: str, context: list) -> list[dict]:
    terms = query.lower().split()
    results = []
    for title, sentences in context:
        text = (title + " " + " ".join(sentences)).lower()
        if any(t in text for t in terms):
            results.append({"title": title, "sentences": sentences})
    return results


def execute_read_paragraph(title: str, context: list) -> str | None:
    for ctx_title, sentences in context:
        if ctx_title == title:
            return " ".join(sentences)
    return None


def execute_tool(tool_name: str, tool_input: dict, context: list) -> str:
    if tool_name == "search_paragraphs":
        results = execute_search_paragraphs(tool_input["query"], context)
        return json.dumps(results, indent=2) if results else "No paragraphs found matching your query."
    elif tool_name == "read_paragraph":
        result = execute_read_paragraph(tool_input["title"], context)
        return result if result else f"No paragraph found with title '{tool_input['title']}'."
    return f"Unknown tool: {tool_name}"


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(question: str, context: list) -> dict:
    """
    Run the QA agent on a question with given context paragraphs.

    Returns:
        {"answer": str, "trajectory": list, "token_usage": int}
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    trajectory   = []
    total_tokens = 0
    max_turns    = 10

    for turn in range(max_turns):
        with langfuse.start_as_current_generation(
            name=f"agent-turn-{turn + 1}",
            model=MODEL,
            input=[m.content for m in messages],
        ) as generation:
            response = llm.invoke(messages)
            generation.update(
                output=response.content,
                usage_details={
                    "input":  getattr(response.usage_metadata, "input_tokens",  0),
                    "output": getattr(response.usage_metadata, "output_tokens", 0),
                },
            )

        total_tokens += (
            getattr(response.usage_metadata, "input_tokens",  0) +
            getattr(response.usage_metadata, "output_tokens", 0)
        )

        if not response.tool_calls:
            return {
                "answer":      response.content or "Unable to determine the answer.",
                "trajectory":  trajectory,
                "token_usage": total_tokens,
            }

        messages.append(response)

        for tc in response.tool_calls:
            tool_output = execute_tool(tc["name"], tc["args"], context)
            trajectory.append({
                "tool":   tc["name"],
                "input":  tc["args"],
                "output": tool_output,
            })
            messages.append(ToolMessage(content=tool_output, tool_call_id=tc["id"]))

    return {
        "answer":      "Unable to determine the answer.",
        "trajectory":  trajectory,
        "token_usage": total_tokens,
    }
