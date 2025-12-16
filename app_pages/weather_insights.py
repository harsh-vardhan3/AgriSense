"""
Weather insights page module
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from ui.components import render_header, render_subheader
from config.constants import LOCATIONS
from services.weather_service import WeatherService


def render_weather_insights_page(weather_service):
    """
    Render the weather insights page
    
    Args:
        weather_service: WeatherService instance
    """
    render_header("Weather Insights")
    
    st.markdown("""
    Get real-time weather forecasts for your location to plan farming activities effectively.
    """)
    
    # Location selection
    location = st.selectbox("Select your location", [loc["name"] for loc in LOCATIONS])
    
    if st.button("Get Weather Forecast"):
        with st.spinner("Fetching weather data..."):
            forecast = weather_service.get_weather_forecast(location)
        
        if forecast:
            st.session_state.weather_forecast = forecast
            st.session_state.weather_location = location

    if 'weather_forecast' in st.session_state:
        st.markdown(f"<h3 class='sub-header'>5-Day Weather Forecast for {st.session_state.weather_location}</h3>", unsafe_allow_html=True)
        
        # Display weather data in a table
        df = pd.DataFrame(st.session_state.weather_forecast)
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(df, use_container_width=True)
        
        # Plot temperature trend
        st.markdown("<h4 class='sub-header'>Temperature Trend</h4>", unsafe_allow_html=True)
        fig = px.line(
            df,
            x="date",
            y="temp",
            title="Temperature Over Time",
            labels={"date": "Date", "temp": "Temperature (°C)"},
            markers=True
        )
        st.plotly_chart(fig)
        
        # Plot humidity trend
        st.markdown("<h4 class='sub-header'>Humidity Trend</h4>", unsafe_allow_html=True)
        fig = px.line(
            df,
            x="date",
            y="humidity",
            title="Humidity Over Time",
            labels={"date": "Date", "humidity": "Humidity (%)"},
            markers=True
        )
        st.plotly_chart(fig)
        
        # Weather icons and descriptions
        st.markdown("<h4 class='sub-header'>Weather Conditions</h4>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i, row in df.iterrows():
            with cols[i % 5]:
                st.image(f"http://openweathermap.org/img/wn/{row['icon']}@2x.png", width=50)
                st.write(f"{row['date']}")
                st.write(f"{row['weather']}")
                st.write(f"{row['temp']}°C")
