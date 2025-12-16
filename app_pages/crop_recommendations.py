"""
Crop recommendations page module
"""
import streamlit as st
from ui.components import render_header, render_subheader
from crop_recc.logistic import predict_closest_crop


def render_crop_recommendations_page():
    """Render the crop recommendations page"""
    render_header("Crop Recommendations System")
    
    st.markdown("""
    Predict the best crop to grow based on soil parameters.
    """)
    
    # Input fields
    with st.form("crop_prediction_form"):
        nitrogen = st.number_input("Nitrogen (kg/hectare)", min_value=20, max_value=120, value=50)
        phospho = st.number_input("Phosphorus (kg/hectare)", min_value=20, max_value=120, value=50)
        potass = st.number_input("Potassium (kg/hectare)", min_value=20, max_value=120, value=50)
        soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
        soil_moisture = st.number_input("Soil Moisture (%)", min_value=0, max_value=100, value=50)
        
        submitted = st.form_submit_button("Predict Best Crop")
    
    if submitted:
        # Store the input data in session state
        st.session_state.user_input = {
            "nitrogen": nitrogen,
            "phospho": phospho,
            "potass": potass,
            "soil_ph": soil_ph,
            "soil_moisture": soil_moisture
        }
        
        # Call the prediction program
        predicted_crop = predict_closest_crop(nitrogen, phospho, potass, soil_ph, soil_moisture)
        st.session_state.predicted_crop = predicted_crop
    
    # Display the predicted crop
    if 'predicted_crop' in st.session_state:
        render_subheader("Predicted Best Crop")
        
        # Display the predicted crop
        st.markdown(f"""
        <div style='padding: 15px; border-radius: 5px; margin-bottom: 1rem;'>
            <h4>Best Crop to Grow: {st.session_state.predicted_crop}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Display additional recommendations
        st.markdown("<h4>Recommendations:</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Ensure proper irrigation and drainage.
        - Use fertilizers based on soil nutrient levels.
        - Monitor crop health regularly.
        """)
