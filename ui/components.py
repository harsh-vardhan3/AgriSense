"""
UI components module for reusable UI elements
"""
import streamlit as st
import datetime


def render_sidebar():
    """Render the sidebar navigation"""
    st.sidebar.markdown("# AgriSense")
    page = st.sidebar.selectbox(
        "Select a service:",
        ["Home", "Disease Detection", "Outbreak Prediction System",
         "Weather Insights", "Crop Recommendations", "Soil Distribution"]
    )
    return page


def render_header(title):
    """Render a page header"""
    st.markdown(f"<h1 class='main-header'>{title}</h1>", unsafe_allow_html=True)


def render_subheader(title):
    """Render a subheader"""
    st.markdown(f"<h3 class='sub-header'>{title}</h3>", unsafe_allow_html=True)


def render_current_conditions():
    """Render current conditions box"""
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("### Current Conditions")
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    st.write(f"**Date:** {current_date}")
    st.write("**Overall Disease Risk:** Moderate")
    st.write("**Weather:** Partly Cloudy, 28°C")
    st.markdown("</div>", unsafe_allow_html=True)


def render_prevention_tips():
    """Render prevention tips box"""
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("""
    **Prevention Tips:**
    - Practice crop rotation
    - Maintain proper spacing between plants
    - Ensure adequate drainage in fields
    - Use resistant varieties when available
    - Apply appropriate fungicides preventatively during high-risk periods
    """)
    st.markdown("</div>", unsafe_allow_html=True)
