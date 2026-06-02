from app.agents.graph.state import AgentState


def planner_node(state: AgentState) -> dict:
    """
    Decomposes the user's financial question into focused retrieval sub-tasks.
    Loads prompt from: app/prompts/agents/planner.yml
    """
    # TODO: load planner.yml prompt, call LLM, parse plan
    raise NotImplementedError
