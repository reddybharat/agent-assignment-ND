from src.graphs.type import RAGAgentState
from src.utils.openweather import OpenWeatherService

def weather_node(state: RAGAgentState) -> RAGAgentState:
    """
    Node responsible for getting the weather information for the user.
    """
    # Check if location is null or empty
    if state["location"] is None or state["location"].strip() == "":
        state["answer"] = "Couldn't determine location"
        state["status"] = "WeatherNodeCompleted"
        return state
    
    try:
        service = OpenWeatherService()
        # Step 1: Use Geocoding API to get coordinates from location name
        geocoded = service.geocode_location(state["location"])
        lat, lon = geocoded["lat"], geocoded["lon"]

        # Step 2: Use Current Weather API to get weather data using coordinates
        weather_data = service.get_weather(lat, lon)
        response_text = f"The weather in {weather_data['location']} is {weather_data['description']} with a temperature of {weather_data['temperature']}°C."
        state["answer"] = response_text

    except Exception as e:
        state["answer"] = f"Error getting weather data: {str(e)}"
        
    state["status"] = "WeatherNodeCompleted"
    return state