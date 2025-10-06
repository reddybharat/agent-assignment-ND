from src.graphs.type import RAGAgentState
from src.utils.retriever import Retriever

def retriever_node(state: RAGAgentState) -> RAGAgentState:
    """
    Node responsible for retrieving the relevant information from the source material.
    """
    try:
        if not state.get("query"):
            state["error"] = "Query is required but not provided"
        else:
            retriever = Retriever()
            response = retriever.generate_response(state["query"])
            state["answer"] = response
        
    except Exception as e:
        state["answer"] = f"Unexpected error: {str(e)}"
    
    
    state["status"] = "RetrieverNodeCompleted"
    return state