import os
import requests
from typing import Optional, Dict, Any


class OpenWeatherService:
    """Service class for handling OpenWeatherMap API calls including geocoding and weather data."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenWeather API key is required. Set OPENWEATHER_API_KEY environment variable.")
        
        self.base_url = "https://api.openweathermap.org"
        self.geocoding_url = f"{self.base_url}/geo/1.0/direct"
        self.weather_url = f"{self.base_url}/data/2.5/weather"
    
    def geocode_location(self, location: str) -> Dict[str, Any]:
        """
        Convert location name to coordinates using OpenWeatherMap Geocoding API.
        
        Args:
            location: Location name (e.g., "London", "New York", "Tokyo")
            
        Returns:
            Dict containing coordinates and location info
        """
        params = {
            "q": location,
            "limit": 1,
            "appid": self.api_key
        }
        
        try:
            response = requests.get(self.geocoding_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                raise ValueError(f"Location '{location}' not found")
            
            location_data = data[0]
            return {
                "name": location_data.get("name", location),
                "country": location_data.get("country", ""),
                "state": location_data.get("state", ""),
                "lat": location_data.get("lat"),
                "lon": location_data.get("lon")
            }
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to geocode location: {str(e)}")
    
    def get_weather(self, lat: float, lon: float, units: str = "metric") -> Dict[str, Any]:
        """
        Get current weather data for given coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            units: Temperature units (metric, imperial, or standard)
            
        Returns:
            Dict containing weather data
        """
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = requests.get(self.weather_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "location": data.get("name", "Unknown"),
                "country": data.get("sys", {}).get("country", ""),
                "temperature": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "humidity": data.get("main", {}).get("humidity"),
                "pressure": data.get("main", {}).get("pressure"),
                "description": data.get("weather", [{}])[0].get("description", ""),
                "main_weather": data.get("weather", [{}])[0].get("main", ""),
                "wind_speed": data.get("wind", {}).get("speed"),
                "wind_direction": data.get("wind", {}).get("deg"),
                "visibility": data.get("visibility"),
                "cloudiness": data.get("clouds", {}).get("all"),
                "units": units
            }
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get weather data: {str(e)}")


