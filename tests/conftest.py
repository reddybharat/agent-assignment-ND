"""
Simple pytest configuration and basic fixtures.
"""
import pytest
import os
from unittest.mock import patch


@pytest.fixture
def test_api_key():
    """Simple test API key."""
    return "test_api_key_12345"


@pytest.fixture
def sample_weather_data():
    """Basic weather data for testing."""
    return {
        "location": "London",
        "temperature": 15.5,
        "description": "overcast clouds",
        "humidity": 75
    }


@pytest.fixture
def sample_rag_state():
    """Basic RAG state for testing."""
    return {
        "query": "What is the weather in London?",
        "answer": "",
        "is_weather_query": True,
        "location": "London"
    }


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up basic test environment variables."""
    with patch.dict(os.environ, {
        "OPENWEATHER_API_KEY": "test_key",
        "QDRANT_CLOUD_URL": "https://test.com",
        "QDRANT_API_KEY": "test_key",
        "GOOGLE_API_KEY": "test_key"
    }):
        yield
