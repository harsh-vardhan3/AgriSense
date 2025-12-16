"""
Home page module
"""
import streamlit as st
import datetime
from utils.helpers import set_bg_image
from ui.components import render_header, render_current_conditions
from config.constants import HOME_IMAGE_PATH, BACKGROUND_IMAGE_PATH


def render_home_page():
    """Render the home page"""
    set_bg_image()
    render_header("AgriSense- AI Precision Farming Assistant for Smallholder farmers")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to AI-Powered Precision Farming System AgriSense
        
        This platform helps farmers identify plant diseases, predict potential outbreaks, 
        monitor environmental conditions, and receive personalized recommendations to
        improve crop health and yield.
        
        **Key Features:**
        - 📱 Plant disease detection from smartphone images
        - 🔮 Disease outbreak prediction and early warnings
        - 🌱 Soil health monitoring and recommendations
        - 🌤️ Weather insights and alerts
        - 📊 Crop performance analytics
        
        Select a service from the sidebar to begin.
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.image(HOME_IMAGE_PATH)
        render_current_conditions()
    
    # Map of Manipur showing agricultural insights
    st.image(BACKGROUND_IMAGE_PATH, width=408)
