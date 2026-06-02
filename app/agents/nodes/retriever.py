from app.agents.graph.state import AgentState


def retriever_node(state: AgentState) -> dict:
    """
    Runs semantic search against the vector store for each plan sub-task.
    On re-entry: uses state['retrieval_gap'] to refine the query.
    Increments iteration_count each pass.
    """
    # TODO: vector store search, merge chunks, increment iteration_count
    raise NotImplementedError
