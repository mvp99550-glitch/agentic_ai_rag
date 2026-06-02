from app.agents.graph.state import AgentState


def generator_node(state: AgentState) -> dict:
    """
    Synthesizes the final answer from validated retrieved chunks.
    Every claim must map to a source chunk (enforced by output guardrails).
    Loads prompt from: app/prompts/agents/generator.yml
    """
    # TODO: load generator.yml prompt, call LLM, extract sources, compute confidence
    raise NotImplementedError
