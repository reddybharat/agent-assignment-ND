from src.graphs.type import RAGAgentState
from src.utils.ingest_pdf import IngestPDF

def ingestor_node(state: RAGAgentState) -> RAGAgentState:
    """
    Node responsible for ingesting the source material into the vector store.
    """
    ingestor = IngestPDF()
    ingestor.run_ingestion_pipeline(state["files_uploaded"])
    state["status"] = "IngestorNodeCompleted"
    
    return state