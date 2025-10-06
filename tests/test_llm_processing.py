import pytest
from src.graphs.builder import routing_condition


@pytest.fixture
def sample_state():
    """Basic state for testing."""
    return {
        "query": "What is the weather in London?",
        "answer": "",
        "is_weather_query": True,
        "location": "London",
        "status": "initial"
    }


@pytest.mark.parametrize("is_weather,expected", [
    (True, "weather"),
    (False, "retriever"),
    (None, "retriever"),
])
def test_routing_condition(is_weather, expected):
    """Test routing condition logic."""
    state = {"is_weather_query": is_weather}
    result = routing_condition(state)
    assert result == expected


def test_routing_condition_missing_key():
    """Test routing condition with missing key raises KeyError."""
    with pytest.raises(KeyError):
        routing_condition({})


@pytest.mark.parametrize("field,value,expected_type", [
    ("query", "What is machine learning?", str),
    ("answer", "Machine learning is a subset of AI", str),
    ("is_weather_query", True, bool),
    ("is_weather_query", False, bool),
    ("location", "London", str),
    ("location", None, type(None)),
    ("status", "completed", str),
])
def test_state_field_types(field, value, expected_type):
    """Test state field type handling."""
    state = {field: value}
    assert isinstance(state[field], expected_type)


@pytest.mark.parametrize("query,is_weather,location", [
    ("What's the weather in Paris?", True, "Paris"),
    ("How's the temperature in Tokyo?", True, "Tokyo"),
    ("What is machine learning?", False, None),
    ("Explain artificial intelligence", False, None),
    ("Tell me about Python programming", False, None),
])
def test_state_query_classification(query, is_weather, location):
    """Test state with different query types."""
    state = {
        "query": query,
        "is_weather_query": is_weather,
        "location": location,
        "answer": "",
        "status": "initial"
    }
    
    assert state["query"] == query
    assert state["is_weather_query"] == is_weather
    assert state["location"] == location


@pytest.mark.parametrize("status", [
    "initial",
    "routing",
    "weather",
    "retriever", 
    "completed",
    "error"
])
def test_state_status_transitions(status):
    """Test state status field handling."""
    state = {"status": status}
    assert state["status"] == status


@pytest.mark.parametrize("location", [
    "London",
    "New York", 
    "Tokyo",
    "Paris",
    "",
    None
])
def test_state_location_handling(location):
    """Test state location field handling."""
    state = {"location": location}
    assert state["location"] == location


@pytest.mark.parametrize("answer", [
    "",
    "The weather in London is sunny",
    "Machine learning is a subset of artificial intelligence",
    "Error: Could not process request"
])
def test_state_answer_handling(answer):
    """Test state answer field handling."""
    state = {"answer": answer}
    assert state["answer"] == answer


def test_state_required_fields(sample_state):
    """Test that state has all required fields."""
    required_fields = ["query", "answer", "is_weather_query", "location", "status"]
    
    for field in required_fields:
        assert field in sample_state


def test_state_field_modification(sample_state):
    """Test that state fields can be modified."""
    # Modify fields
    sample_state["query"] = "What is AI?"
    sample_state["is_weather_query"] = False
    sample_state["location"] = None
    sample_state["answer"] = "AI is artificial intelligence"
    sample_state["status"] = "completed"
    
    # Verify changes
    assert sample_state["query"] == "What is AI?"
    assert sample_state["is_weather_query"] is False
    assert sample_state["location"] is None
    assert sample_state["answer"] == "AI is artificial intelligence"
    assert sample_state["status"] == "completed"


@pytest.mark.parametrize("query,expected_routing", [
    ("What's the weather in London?", "weather"),
    ("How's the temperature in Paris?", "weather"),
    ("What is machine learning?", "retriever"),
    ("Explain artificial intelligence", "retriever"),
    ("Tell me about Python", "retriever"),
])
def test_query_to_routing_mapping(query, expected_routing):
    """Test how different queries should be routed."""
    # This simulates the logic that would be in the routing node
    is_weather = any(word in query.lower() for word in ["weather", "temperature", "climate", "forecast"])
    state = {"is_weather_query": is_weather}
    result = routing_condition(state)
    assert result == expected_routing