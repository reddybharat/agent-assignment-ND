"""
Simple test cases for Retriever functionality.
"""
import pytest
from unittest.mock import Mock, patch
from src.utils.retriever import Retriever


def test_retriever_init_without_api_key():
    """Test that retriever raises error without API key."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError):
            Retriever()


@patch('src.utils.retriever.QdrantClient')
@patch('src.utils.retriever.SentenceTransformer')
@patch('src.utils.retriever.QdrantVectorStore')
@patch('src.utils.retriever.GoogleGenerativeAI')
def test_retriever_init_success(mock_llm, mock_vector_store, mock_transformer, mock_client):
    """Test successful retriever initialization."""
    retriever = Retriever()
    assert retriever is not None


@patch('src.utils.retriever.QdrantClient')
@patch('src.utils.retriever.SentenceTransformer')
@patch('src.utils.retriever.QdrantVectorStore')
@patch('src.utils.retriever.GoogleGenerativeAI')
def test_retrieve_documents(mock_llm, mock_vector_store, mock_transformer, mock_client):
    """Test document retrieval."""
    # Mock documents
    doc1 = Mock()
    doc1.page_content = "Sample content 1"
    doc1.metadata = {"source": "test.pdf", "page": 1}
    
    doc2 = Mock()
    doc2.page_content = "Sample content 2"
    doc2.metadata = {"source": "test.pdf", "page": 2}
    
    # Mock vector store
    mock_vector_store_instance = Mock()
    mock_vector_store_instance.similarity_search.return_value = [doc1, doc2]
    mock_vector_store.return_value = mock_vector_store_instance
    
    retriever = Retriever()
    results = retriever.retrieve("test query", k=2)
    
    assert len(results) == 2
    assert results[0].page_content == "Sample content 1"


@patch('src.utils.retriever.QdrantClient')
@patch('src.utils.retriever.SentenceTransformer')
@patch('src.utils.retriever.QdrantVectorStore')
@patch('src.utils.retriever.GoogleGenerativeAI')
def test_generate_response(mock_llm, mock_vector_store, mock_transformer, mock_client):
    """Test response generation."""
    # Mock documents
    doc1 = Mock()
    doc1.page_content = "Sample content"
    doc1.metadata = {"source": "test.pdf", "page": 1}
    
    # Mock vector store and LLM
    mock_vector_store_instance = Mock()
    mock_vector_store_instance.similarity_search.return_value = [doc1]
    mock_vector_store.return_value = mock_vector_store_instance
    
    mock_llm_instance = Mock()
    mock_llm_instance.invoke.return_value = "Generated response"
    mock_llm.return_value = mock_llm_instance
    
    retriever = Retriever()
    response = retriever.generate_response("test query")
    
    assert response == "Generated response"
