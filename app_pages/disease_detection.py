"""
Disease detection page module
"""
import streamlit as st
from PIL import Image
from ui.components import render_header, render_subheader, render_prevention_tips
from services.disease_detection import DiseaseDetector


def render_disease_detection_page(detector):
    """
    Render the disease detection page
    
    Args:
        detector: DiseaseDetector instance
    """
    render_header("Plant Disease Detection")
    
    st.markdown("""
    Upload an image of your plant to identify diseases and get treatment recommendations.
    The AI model can analyze leaves, stems, and full plants to detect various diseases affecting crops in Manipur.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])
        camera_input = st.camera_input("Or take a photo")
        
        if st.button("Submit for Analysis"):
            if uploaded_file is not None or camera_input is not None:
                image_file = uploaded_file if uploaded_file is not None else camera_input
                image = Image.open(image_file)
                
                with st.spinner("Analyzing image..."):
                    results = detector.detect_disease(image)
                
                st.session_state.analysis_results = results
                st.session_state.analyzed_image = image

    with col2:
        if 'analysis_results' in st.session_state:
            st.image(st.session_state.analyzed_image, caption=f"Uploaded crop image")
        
            render_subheader("Analysis Results")
        
            main_disease = st.session_state.analysis_results[0]
        
            st.markdown(f"""
            <div style='padding: 15px; border-radius: 5px;'>
                <h4>Detected: {main_disease['name']}</h4>
                <p><strong>Description:</strong> {main_disease['description']}</p>
                <p><strong>Recommended Treatment:</strong> {main_disease['treatment']}</p>
            </div>
            """, unsafe_allow_html=True)
        
            # Display supplement information
            st.markdown("<h4>Recommended Supplement:</h4>", unsafe_allow_html=True)
            st.markdown(f"**Name:** {main_disease['supplement_name']}")
            st.image(main_disease['supplement_image_url'], caption=main_disease['supplement_name'])
            st.markdown(f"**Buy Link:** [Purchase Here]({main_disease['supplement_buy_link']})")

            render_prevention_tips()
        else:
            st.image("https://via.placeholder.com/400x300?text=Plant+Image+Preview")
            st.info("Upload an image to see disease detection results")
