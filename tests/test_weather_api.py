import pytest
from src.utils.openweather import OpenWeatherService
from dotenv import load_dotenv
load_dotenv()

@pytest.mark.parametrize("location, expected_name, expected_country", [
    ("London", "London", "GB"),
    ("New York", "New York", "US"),
    ("Tokyo", "Tokyo", "JP"),
    ("Paris", "Paris", "FR"),
    ("Sydney", "Sydney", "AU"),
])
def test_geocode_location_valid(location, expected_name, expected_country):
    """Test geocoding valid locations using real OpenWeather API."""
    service = OpenWeatherService()
    result = service.geocode_location(location)
    
    assert isinstance(result, dict)
    assert "name" in result
    assert "country" in result
    assert "lat" in result
    assert "lon" in result
    
    assert result["name"] == expected_name
    assert result["country"] == expected_country
    assert isinstance(result["lat"], (int, float))
    assert isinstance(result["lon"], (int, float))


@pytest.mark.parametrize("invalid_location", [
    "NonExistentCity12345",
    "InvalidLocationXYZ",
    "FakeCity123",
])
def test_geocode_location_invalid(invalid_location):
    """Test geocoding invalid locations should raise ValueError."""
    service = OpenWeatherService()
    
    with pytest.raises(ValueError, match=f"Location '{invalid_location}' not found"):
        service.geocode_location(invalid_location)


@pytest.mark.parametrize("lat, lon, expected_location, expected_country", [
    (51.5074, -0.1278, "London", "GB"),
    (40.7128, -74.0060, "New York", "US"),
    (48.8566, 2.3522, "Paris", "FR"), 
    (-33.8688, 151.2093, "Sydney", "AU"),
])
def test_get_weather_valid_coordinates(lat, lon, expected_location, expected_country):
    """Test getting weather data for valid coordinates using real OpenWeather API."""
    service = OpenWeatherService()
    result = service.get_weather(lat, lon)
    
    # Verify response structure
    assert isinstance(result, dict)
    required_fields = [
        "location", "country", "temperature", "feels_like", 
        "humidity", "pressure", "description", "main_weather",
        "wind_speed", "wind_direction", "visibility", "cloudiness", "units"
    ]
    
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    # Verify data integrity
    assert result["location"] == expected_location
    assert result["country"] == expected_country
    assert isinstance(result["temperature"], (int, float))
    assert isinstance(result["humidity"], (int, float))
    assert isinstance(result["description"], str)
    assert result["units"] == "metric"


@pytest.mark.parametrize("invalid_lat, invalid_lon", [
    (999.0, 999.0),  
    (-999.0, -999.0),       
])
def test_get_weather_invalid_coordinates(invalid_lat, invalid_lon):
    """Test getting weather data for invalid coordinates should raise Exception with 400 status."""
    service = OpenWeatherService()
    
    with pytest.raises(Exception, match="400 Client Error: Bad Request"):
        service.get_weather(invalid_lat, invalid_lon)
