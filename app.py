"""
AgriSense - AI Precision Farming Assistant for Smallholder Farmers
Modular main application entry point
"""
import streamlit as st
import warnings

# Import services
from services import DiseaseDetector, WeatherService

# Import UI components
from ui.components import render_sidebar
from utils.helpers import apply_custom_css

# Import pages
from app_pages import (
    render_home_page,
    render_disease_detection_page,
    render_outbreak_prediction_page,
    render_crop_recommendations_page,
    render_weather_insights_page,
    render_soil_distribution_page
)

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Precision Farming for smallscale farmers",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Initialize services
@st.cache_resource
def get_disease_detector():
    """Initialize and cache disease detector"""
    return DiseaseDetector()

@st.cache_resource
def get_weather_service():
    """Initialize and cache weather service"""
    return WeatherService()

# Get service instances
disease_detector = get_disease_detector()
weather_service = get_weather_service()

# Render sidebar navigation
page = render_sidebar()

# Route to appropriate page
if page == "Home":
    render_home_page()

elif page == "Disease Detection":
    render_disease_detection_page(disease_detector)

elif page == "Outbreak Prediction System":
    render_outbreak_prediction_page()

elif page == "Crop Recommendations":
    render_crop_recommendations_page()

elif page == "Weather Insights":
    render_weather_insights_page(weather_service)

elif page == "Soil Distribution":
    render_soil_distribution_page()