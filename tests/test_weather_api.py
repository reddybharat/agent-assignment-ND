"""
Simple test cases for OpenWeather API functionality.
"""
import pytest
import requests
from unittest.mock import Mock, patch
from src.utils.openweather import OpenWeatherService


def test_init_with_api_key(test_api_key):
    """Test initialization with API key."""
    service = OpenWeatherService(api_key=test_api_key)
    assert service.api_key == test_api_key


def test_init_without_api_key():
    """Test initialization without API key raises error."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError):
            OpenWeatherService()


@patch('requests.get')
def test_geocode_location_success(mock_get, test_api_key):
    """Test successful geocoding."""
    # Mock API response
    mock_response = Mock()
    mock_response.json.return_value = [{
        "name": "London",
        "country": "GB",
        "lat": 51.5074,
        "lon": -0.1278
    }]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    service = OpenWeatherService(api_key=test_api_key)
    result = service.geocode_location("London")
    
    assert result["name"] == "London"
    assert result["lat"] == 51.5074


@patch('requests.get')
def test_geocode_location_not_found(mock_get, test_api_key):
    """Test geocoding when location is not found."""
    mock_response = Mock()
    mock_response.json.return_value = []
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    service = OpenWeatherService(api_key=test_api_key)
    
    with pytest.raises(ValueError):
        service.geocode_location("Unknown City")


@patch('requests.get')
def test_get_weather_success(mock_get, test_api_key):
    """Test successful weather data retrieval."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "name": "London",
        "main": {"temp": 15.5, "humidity": 75},
        "weather": [{"description": "overcast clouds"}]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    service = OpenWeatherService(api_key=test_api_key)
    result = service.get_weather(51.5074, -0.1278)
    
    assert result["location"] == "London"
    assert result["temperature"] == 15.5
    assert result["description"] == "overcast clouds"


@patch('requests.get')
def test_get_weather_api_error(mock_get, test_api_key):
    """Test weather retrieval when API returns an error."""
    mock_get.side_effect = requests.exceptions.RequestException("API Error")
    
    service = OpenWeatherService(api_key=test_api_key)
    
    with pytest.raises(Exception):
        service.get_weather(51.5074, -0.1278)
