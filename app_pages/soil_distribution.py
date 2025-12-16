"""
Soil distribution page module
"""
import streamlit as st
from ui.components import render_header
from config.constants import SOIL_IMAGE_PATH


def render_soil_distribution_page():
    """Render the soil distribution page"""
    render_header("Soil Distribution")
    
    st.markdown("""
    Explore the soil distribution across different regions of India.
    """)
    
    # Display the soil distribution image
    st.image(SOIL_IMAGE_PATH, caption="Soil Distribution Map of India")
    
    # Add a download button for the soil distribution map
    with open(SOIL_IMAGE_PATH, "rb") as file:
        btn = st.download_button(
            label="Download Soil Distribution Map",
            data=file,
            file_name="soil_distribution_map.png",
            mime="image/png"
        )
