"""
Outbreak prediction page module
"""
import streamlit as st
import plotly.graph_objects as go
from ui.components import render_header, render_subheader
from config.constants import CROP_TYPES, DISEASES, CITIES, MONTHS, OUTBREAK_PATHS
from outbreak_prediction.ScorePredictor import DiseaseOutbreakPredictor


def render_outbreak_prediction_page():
    """Render the outbreak prediction page"""
    render_header("Outbreak Prediction System")
    
    st.markdown("""
    Predict disease outbreak risks based on environmental factors, crop type, and location.
    """)
    
    # Input fields
    with st.form("outbreak_prediction_form"):
        month = st.selectbox("Select Month", MONTHS)
        crop = st.selectbox("Select Crop Type", CROP_TYPES)
        disease = st.selectbox("Select Disease", DISEASES)
        city = st.selectbox("Select City", CITIES)
        
        submitted = st.form_submit_button("Predict Risk Score")
    
    if submitted:
        # Store the input data in session state
        st.session_state.user_input = {
            "month": month,
            "crop": crop,
            "disease": disease,
            "city": city
        }

        predictor = DiseaseOutbreakPredictor(
            future_weather_path=OUTBREAK_PATHS["future_weather"],
            past_weather_path=OUTBREAK_PATHS["past_weather"],
            crop_susceptibility_path=OUTBREAK_PATHS["crop_susceptibility"],
            disease_virality_path=OUTBREAK_PATHS["disease_virality"],
            city_weightage_path=OUTBREAK_PATHS["city_weightage"],
            month_weightage_path=OUTBREAK_PATHS["month_weightage"]
        )
        
        risk_score = predictor.calculate_risk_score(
            month=st.session_state.user_input["month"],
            crop=st.session_state.user_input["crop"],
            disease=st.session_state.user_input["disease"],
            city=st.session_state.user_input["city"]
        )

        st.session_state.risk_score = risk_score

    if 'risk_score' in st.session_state:
        render_subheader("Predicted Risk Score")
        
        # Display the risk score as a meter bar
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=st.session_state.risk_score,
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#43a047"},
                'steps': [
                    {'range': [0, 30], 'color': "#c8e6c9"},  # Low risk
                    {'range': [30, 70], 'color': "#fff9c4"},  # Moderate risk
                    {'range': [70, 100], 'color': "#ffcdd2"}  # High risk
                ]
            }
        ))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig)
        
        # Display risk level based on score
        if st.session_state.risk_score < 30:
            st.success("Low Risk: No significant outbreak expected.")
        elif st.session_state.risk_score < 70:
            st.warning("Moderate Risk: Monitor conditions closely.")
        else:
            st.error("High Risk: Take preventive measures immediately.")
