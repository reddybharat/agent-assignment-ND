from src.graphs.type import RAGAgentState
from langchain_google_genai import GoogleGenerativeAI
from src.utils.prompts import WEATHER_CLASSIFICATION_PROMPT
import json

from dotenv import load_dotenv
load_dotenv()

def routing_node(state: RAGAgentState) -> RAGAgentState:
    """
    Node responsible for routing the user query to the appropriate node.
    """
    # Initialize Gemini LLM
    llm = GoogleGenerativeAI(model="gemini-2.0-flash")
    
    # Classify if the query is about weather and extract location
    classification_prompt = WEATHER_CLASSIFICATION_PROMPT.format(query=state["query"])
    classification_response = llm.invoke(classification_prompt)
    
    try:
        # Extract JSON from markdown code blocks
        response_text = classification_response.strip().strip("```").strip("json")
        result = json.loads(response_text)
        state["is_weather_query"] = result.get("is_weather", False)
        state["location"] = result.get("location", None)

    except json.JSONDecodeError:
        state["is_weather_query"] = False
        state["location"] = None
        state["answer"] = "Error parsing JSON response"
    
    state["status"] = "RoutingNodeCompleted"
    return state