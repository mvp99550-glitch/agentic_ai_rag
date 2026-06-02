from app.agents.graph.state import AgentState

MAX_RETRIEVAL_ITERATIONS = 3


def route_after_reasoner(state: AgentState) -> str:
    """
    After the reasoner validates retrieved context:
    - If context is sufficient OR we've hit the iteration cap → generate
    - Otherwise → retrieve again with the identified gap
    """
    if state["iteration_count"] >= MAX_RETRIEVAL_ITERATIONS:
        return "generate"
    if state["context_sufficient"]:
        return "generate"
    return "retrieve"
