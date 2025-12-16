"""
Weather service module for fetching weather data
"""
import requests
import streamlit as st
from config.constants import API_KEY2, LOCATIONS


class WeatherService:
    """Weather service class for fetching weather forecasts"""
    
    def __init__(self):
        """Initialize the weather service"""
        self.api_key = API_KEY2
        self.locations = LOCATIONS
    
    def get_weather_forecast(self, location):
        """
        Fetch real-time weather forecast data using OpenWeatherMap API
        
        Args:
            location: Name of the location
            
        Returns:
            list: List of dictionaries containing weather forecast data
        """
        # Get latitude and longitude for the selected location
        loc_data = next((loc for loc in self.locations if loc["name"] == location), None)
        if not loc_data:
            st.error("Location not found. Please select a valid location.")
            return None
        
        lat, lon = loc_data["lat"], loc_data["lon"]
        
        # API endpoint for 5-day weather forecast
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        response = requests.get(url)
        
        # Debug: Print the API response
        print("API Response Status Code:", response.status_code)
        print("API Response Content:", response.text)
        
        if response.status_code == 200:
            data = response.json()
            forecast = []
            for item in data["list"]:
                forecast.append({
                    "date": item["dt_txt"],
                    "temp": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                    "wind_speed": item["wind"]["speed"],
                    "weather": item["weather"][0]["description"],
                    "icon": item["weather"][0]["icon"]
                })
            return forecast
        else:
            st.error(f"Failed to fetch weather data. Status Code: {response.status_code}")
            return None
