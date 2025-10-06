import pytest
from src.utils.retriever import Retriever
from dotenv import load_dotenv
load_dotenv()


def test_retriever_init_without_api_key():
    """Test that retriever raises error without API key."""
    import os
    original_key = os.environ.get("QDRANT_API_KEY")
    try:
        # Remove the API key
        if "QDRANT_API_KEY" in os.environ:
            del os.environ["QDRANT_API_KEY"]
        
        with pytest.raises(ValueError, match="QDRANT_API_KEY environment variable is not set. Please set it with your QDrant Cloud API key."):
            Retriever()
    finally:
        # Restore the original API key
        if original_key:
            os.environ["QDRANT_API_KEY"] = original_key


def test_retriever_init_success():
    """Test successful retriever initialization."""
    retriever = Retriever()
    assert retriever is not None
    assert retriever.collection_name == "uploaded-pdfs"
    assert retriever.client is not None
    assert retriever.llm is not None


@pytest.mark.parametrize("query, k", [
    ("machine learning", 3),
    ("artificial intelligence", 5),
    ("data science", 2),
])
def test_retrieve_documents(query, k):
    """Test document retrieval with different queries and k values."""
    retriever = Retriever()
    results = retriever.retrieve(query, k=k)
    
    # Verify results structure
    assert isinstance(results, list)
    assert len(results) <= k
    
    # Check that each result has the expected attributes
    for result in results:
        assert isinstance(result, dict)
        assert 'page_content' in result
        assert 'metadata' in result
        assert isinstance(result['metadata'], dict)


@pytest.mark.parametrize("query", [
    "What is machine learning?",
    "How does artificial intelligence work?",
    "Explain data science concepts",
])
def test_generate_response(query):
    """Test response generation for different queries."""
    retriever = Retriever()
    response = retriever.generate_response(query)
    
    # Verify response is a string
    assert isinstance(response, str)
    assert len(response) > 0


def test_retrieve_with_custom_k():
    """Test retrieve method with custom k parameter."""
    retriever = Retriever()
    
    # Test with k=1
    results_1 = retriever.retrieve("test query", k=1)
    assert len(results_1) <= 1
    
    # Test with k=5
    results_5 = retriever.retrieve("test query", k=5)
    assert len(results_5) <= 5


def test_generate_response_with_custom_k():
    """Test generate_response method with custom k parameter."""
    retriever = Retriever()
    
    response = retriever.generate_response("test query", k=2)
    assert isinstance(response, str)
    assert len(response) > 0
