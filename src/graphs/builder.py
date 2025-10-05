from langgraph.graph import StateGraph, START, END
from src.graphs.nodes.ingestor_node import ingestor_node
from src.graphs.nodes.routing_node import routing_node
from src.graphs.nodes.weather_node import weather_node
from src.graphs.nodes.retriever_node import retriever_node
from src.graphs.type import RAGAgentState

def routing_condition(state: RAGAgentState) -> bool:
    if state["is_weather_query"]:
        return "weather"
    else:
        return "retriever"


def _build_base_graph(state: RAGAgentState) -> StateGraph:

    builder = StateGraph(RAGAgentState)

    builder.add_node("ingestor", ingestor_node)
    builder.add_node("routing", routing_node)
    builder.add_node("weather", weather_node)
    builder.add_node("retriever", retriever_node)

    builder.add_edge(START, "ingestor")
    builder.add_edge("ingestor", "routing")
    builder.add_conditional_edges(
        "routing", routing_condition, {
            "weather": "weather",
            "retriever": "retriever"
        }
    )

    builder.add_edge("weather", END)
    builder.add_edge("retriever", END)

    return builder

def build_graph(state: RAGAgentState) -> StateGraph:
    builder = _build_base_graph(state)
    return builder.compile()