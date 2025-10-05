"""
Simple test cases for LLM processing components.
"""
import pytest
from unittest.mock import Mock, patch
from src.graphs.nodes.routing_node import routing_node
from src.utils.prompts import WEATHER_CLASSIFICATION_PROMPT, RETRIEVER_PROMPT


def test_weather_classification_prompt():
    """Test weather classification prompt formatting."""
    query = "What's the weather in London?"
    formatted_prompt = WEATHER_CLASSIFICATION_PROMPT.format(query=query)
    
    assert query in formatted_prompt
    assert "is_weather" in formatted_prompt
    assert "location" in formatted_prompt


def test_retriever_prompt():
    """Test retriever prompt formatting."""
    context = "Sample context"
    query = "What is the main topic?"
    formatted_prompt = RETRIEVER_PROMPT.format(context=context, query=query)
    
    assert context in formatted_prompt
    assert query in formatted_prompt
    assert "expert assistant" in formatted_prompt


@patch('src.graphs.nodes.routing_node.GoogleGenerativeAI')
def test_routing_node_weather_query(mock_llm_class, sample_rag_state):
    """Test routing node with weather query."""
    # Mock LLM response
    mock_llm_instance = Mock()
    mock_llm_instance.invoke.return_value = '{"is_weather": true, "location": "London"}'
    mock_llm_class.return_value = mock_llm_instance
    
    state = sample_rag_state.copy()
    result = routing_node(state)
    
    assert result["is_weather_query"] is True
    assert result["location"] == "London"


@patch('src.graphs.nodes.routing_node.GoogleGenerativeAI')
def test_routing_node_non_weather_query(mock_llm_class, sample_rag_state):
    """Test routing node with non-weather query."""
    mock_llm_instance = Mock()
    mock_llm_instance.invoke.return_value = '{"is_weather": false, "location": null}'
    mock_llm_class.return_value = mock_llm_instance
    
    state = sample_rag_state.copy()
    state["query"] = "Tell me about Python"
    
    result = routing_node(state)
    
    assert result["is_weather_query"] is False
    assert result["location"] is None


@patch('src.graphs.nodes.routing_node.GoogleGenerativeAI')
def test_routing_node_invalid_json(mock_llm_class, sample_rag_state):
    """Test routing node with invalid JSON response."""
    mock_llm_instance = Mock()
    mock_llm_instance.invoke.return_value = "Invalid JSON"
    mock_llm_class.return_value = mock_llm_instance
    
    result = routing_node(sample_rag_state)
    
    assert result["is_weather_query"] is False
    assert "Error" in result["answer"]
