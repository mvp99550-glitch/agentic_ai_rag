from app.agents.graph.state import AgentState


def reasoner_node(state: AgentState) -> dict:
    """
    Validates whether retrieved chunks sufficiently answer the question.
    Sets context_sufficient=True to proceed, or identifies retrieval_gap to loop.
    Loads prompt from: app/prompts/agents/reasoner.yml
    """
    # TODO: load reasoner.yml prompt, call LLM, set context_sufficient + retrieval_gap
    raise NotImplementedError
