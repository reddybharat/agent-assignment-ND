from typing import TypedDict, List

class RAGAgentState(TypedDict):
    """
    Represents the state of the agent in the state graph.
    """
    query: str
    answer: str
    status: str
    is_weather_query: bool
    location: str