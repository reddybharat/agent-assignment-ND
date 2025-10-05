from typing import TypedDict, List, Annotated, Sequence, Union

class RAGAgentState(TypedDict):
    """
    Represents the state of the agent in the state graph.
    """
    files_uploaded: List[str]
    query: str
    answer: str
    status: str
    is_weather_query: bool
    location: str
