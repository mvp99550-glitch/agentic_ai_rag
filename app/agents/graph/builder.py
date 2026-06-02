from langgraph.graph import StateGraph, END

from app.agents.graph.state import AgentState
from app.agents.graph.edges import route_after_reasoner
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retriever_node
from app.agents.nodes.reasoner import reasoner_node
from app.agents.nodes.generator import generator_node


def build_rag_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("generator", generator_node)

    # Entry point
    graph.set_entry_point("planner")

    # Fixed edges
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "reasoner")

    # Conditional: reasoner decides whether to loop back or generate
    graph.add_conditional_edges(
        "reasoner",
        route_after_reasoner,
        {
            "retrieve": "retriever",
            "generate": "generator",
        },
    )

    graph.add_edge("generator", END)

    return graph.compile()


# Singleton — import this in your API routes
rag_graph = build_rag_graph()
