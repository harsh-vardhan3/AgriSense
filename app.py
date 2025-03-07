import streamlit as st
import pandas as pd
import numpy as np
import cv2
import requests
import matplotlib.pyplot as plt
import datetime
import os
from flask import Flask, redirect, render_template, request
import json
import os
import folium
import torch
import CNN
import torchvision.transforms.functional as TF
import plotly.express as px
from streamlit_folium import folium_static
from PIL import Image
import io
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestClassifier
import warnings
from outbreak_prediction.ScorePredictor import DiseaseOutbreakPredictor
from crop_recc.logistic import predict_closest_crop
from Firebase import call_firebase
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Precision Farming for smallscale farmers",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for the background image
def set_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("images_frontend/c.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            opacity: 1;  /* Adjust opacity here */
        }
        .stApp::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.5);  /* White overlay with 50% opacity */
            z-index: -1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    .section {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #000000;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load disease and supplement information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the trained model
model = CNN.CNN(39)  
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=torch.device('cpu')))  # Use CPU for inference
model.eval()

def detect_disease(image):
    """Predict disease from an image using the trained model."""
    # Resize and preprocess the image
    image = image.resize((224, 224))  # Resize image to match model input size
    input_data = TF.to_tensor(image)  # Convert image to tensor
    input_data = input_data.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)  # Get the predicted class index

    # Get disease details
    disease_name = disease_info['disease_name'][index]
    description = disease_info['description'][index]
    treatment = disease_info['Possible Steps'][index]
    supplement_name = supplement_info['supplement name'][index]
    supplement_image_url = supplement_info['supplement image'][index]
    supplement_buy_link = supplement_info['buy link'][index]

    # Format results
    results = [
        {
            "name": disease_name,
            "treatment": treatment,
            "description": description,
            "supplement_name": supplement_name,
            "supplement_image_url": supplement_image_url,
            "supplement_buy_link": supplement_buy_link
        }
    ]

    return results  



API_KEY2 = "f82b855ceef9e8f9a7fd1ccafda58615"
def get_weather_forecast(location):
    """Fetch real-time weather forecast data using OpenWeatherMap API."""
    # Get latitude and longitude for the selected location
    loc_data = next((loc for loc in LOCATIONS if loc["name"] == location), None)
    if not loc_data:
        st.error("Location not found. Please select a valid location.")
        return None

    lat, lon = loc_data["lat"], loc_data["lon"]

    # API endpoint for 5-day weather forecast
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY2}&units=metric"
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



# Sample locations in Manipur (actual implementation would use a database)
LOCATIONS = [
    {"name": "Imphal East", "lat": 24.8060, "lon": 93.9387},  # Porompat
    {"name": "Imphal West", "lat": 24.8065, "lon": 93.9389},   # Lamphelpat
    {"name": "Thoubal", "lat": 24.6388, "lon": 94.0033},       # Thoubal Town
    {"name": "Bishnupur", "lat": 24.6276, "lon": 93.7739},     # Bishnupur Town
    {"name": "Churachandpur", "lat": 24.3333, "lon": 93.6833}, # Churachandpur Town
    {"name": "Chandel", "lat": 24.3167, "lon": 94.0333},       # Chandel Town
    {"name": "Senapati", "lat": 25.2675, "lon": 94.0667},      # Senapati Town
    {"name": "Ukhrul", "lat": 25.1167, "lon": 94.3667},        # Ukhrul Town
    {"name": "Tamenglong", "lat": 24.9750, "lon": 93.5167},    # Tamenglong Town
    {"name": "Jiribam", "lat": 24.8000, "lon": 93.1167}        # Jiribam Town
]

# Sample crop types common in Manipur
CROP_TYPES = [
    "Apple", "Blueberry", "Cherry", "Corn", "Grape", 
    "Orange", "Peach", "Pepper", "Potato", "Raspberry", 
    "Soybean", "Squash", "Strawberry", "Tomato"
]

# Navigation

st.sidebar.markdown("# AgriSense")
page = st.sidebar.selectbox(
    "Select a service:",
    ["Home", "Disease Detection", "Outbreak Prediction System", "Resource Optimization",
     "Weather Insights", "Crop Recommendations", "Soil Distribution", "Drone Hyperspectral Imagery" ]
)

if page == "Home":
    set_bg_image()
    st.markdown("<h1 class='main-header'>AgriSense- AI Precision Farming Assistant for Smallholder farmers</h1>", unsafe_allow_html=True)
    
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
        - 🚁 IoT based crop recommendation
        - 📊 Crop performance analytics
        
        Select a service from the sidebar to begin.
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.image(r"images_frontend/a.jpg", use_container_width=True)
        
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("### Current Conditions")
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        st.write(f"**Date:** {current_date}")
        st.write("**Overall Disease Risk:** Moderate")
        st.write("**Weather:** Partly Cloudy, 28°C")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Map of Manipur showing agricultural insights
    st.image("images_frontend/c.jpg", width=408)
    

elif page == "Disease Detection":
    st.markdown("<h1 class='main-header'>Plant Disease Detection</h1>", unsafe_allow_html=True)
    
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
                    results = detect_disease(image)
                
                st.session_state.analysis_results = results
                st.session_state.analyzed_image = image

    with col2:
        if 'analysis_results' in st.session_state:
            st.image(st.session_state.analyzed_image, caption=f"Uploaded crop image", use_container_width=True)
        
            st.markdown("<h3 class='sub-header'>Analysis Results</h3>", unsafe_allow_html=True)
        
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
            st.image(main_disease['supplement_image_url'], caption=main_disease['supplement_name'], use_container_width=True)
            st.markdown(f"**Buy Link:** [Purchase Here]({main_disease['supplement_buy_link']})")

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
        else:
            st.image("https://via.placeholder.com/400x300?text=Plant+Image+Preview", use_container_width=True)
            st.info("Upload an image to see disease detection results")
    

elif page == "Outbreak Prediction System":
    st.markdown("<h1 class='main-header'>Outbreak Prediction System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Predict disease outbreak risks based on environmental factors, crop type, and location.
    """)
    
    # Input fields
    with st.form("outbreak_prediction_form"):
        month = st.selectbox("Select Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
        crop = st.selectbox("Select Crop Type", CROP_TYPES)
        disease = st.selectbox("Select Disease", ["Apple Black Rot",
    "Apple Cedar Rust",
    "Apple Scab",
    "Cherry Powdery Mildew",
    "Corn Cercospora Leaf Spot",
    "Northern Corn Leaf Blight",
    "Grape Black Rot",
    "Grapevine Esca",
    "Grape Leaf Blight",
    "Citrus Huanglongbing",
    "Peach Bacterial Spot",
    "Pepper Bacterial Spot",
    "Potato Early Blight",
    "Potato Late Blight",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Mosaic Virus",
    "Tomato Septoria Leaf Spot",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Spider Mite Infestation"])
        city = st.selectbox("Select City", ["Imphal West", "Imphal East", "Churachandpur", "Thoubal", "Senapati"])
        
        submitted = st.form_submit_button("Predict Risk Score")
    
    if submitted:
        # Store the input data in session state or a variable
        st.session_state.user_input = {
            "month": month,
            "crop": crop,
            "disease": disease,
            "city": city
        }

        predictor = DiseaseOutbreakPredictor(
        future_weather_path="outbreak_prediction/processed_FutureWeatherPrediction.csv",
        past_weather_path="outbreak_prediction/processed_PastWeatherData.csv",
        crop_susceptibility_path="outbreak_prediction/DiseaseSusceptibilityOfCrops.csv",
        disease_virality_path="outbreak_prediction/ViralityScoreOfDiseases.csv",
        city_weightage_path="outbreak_prediction/CityWeightage.csv",
        month_weightage_path="outbreak_prediction/MonthWeightage.csv"
        )
        
        risk_score = predictor.calculate_risk_score(
        month=st.session_state.user_input["month"],
        crop=st.session_state.user_input["crop"],
        disease=st.session_state.user_input["disease"],
        city=st.session_state.user_input["city"]
        )

        st.session_state.risk_score = risk_score

        if 'risk_score' in st.session_state:
            st.markdown("<h3 class='sub-header'>Predicted Risk Score</h3>", unsafe_allow_html=True)
            
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
            st.plotly_chart(fig, use_container_width=True)
            
            # Display risk level based on score
            if st.session_state.risk_score < 30:
                st.success("Low Risk: No significant outbreak expected.")
            elif st.session_state.risk_score < 70:
                st.warning("Moderate Risk: Monitor conditions closely.")
            else:
                st.error("High Risk: Take preventive measures immediately.")



elif page == "Crop Recommendations":
    st.markdown("<h1 class='main-header'>Crop Recommendations System</h1>", unsafe_allow_html=True)
    
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
        # Store the input data in session state or a variable
        st.session_state.user_input = {
            "nitrogen": nitrogen,
            "phospho": phospho,
            "potass": potass,
            "soil_ph": soil_ph,
            "soil_moisture": soil_moisture
        }
        
        # Call the prediction program (placeholder function)
        predicted_crop = predict_closest_crop(nitrogen, phospho, potass, soil_ph, soil_moisture)
        st.session_state.predicted_crop = predicted_crop
    
    # Display the predicted crop
    if 'predicted_crop' in st.session_state:
        st.markdown("<h3 class='sub-header'>Predicted Best Crop</h3>", unsafe_allow_html=True)
        
        # Display the predicted crop
        st.markdown(f"""
        <div style='padding: 15px; border-radius: 5px; margin-bottom: 1rem;'>
            <h4>Best Crop to Grow: {st.session_state.predicted_crop}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Display additional recommendations (optional)
        st.markdown("<h4>Recommendations:</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Ensure proper irrigation and drainage.
        - Use fertilizers based on soil nutrient levels.
        - Monitor crop health regularly.
        """)

# Weather Insights Page
elif page == "Weather Insights":
    st.markdown("<h1 class='main-header'>Weather Insights</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Get real-time weather forecasts for your location to plan farming activities effectively.
    """)
    
    # Location selection
    location = st.selectbox("Select your location", [loc["name"] for loc in LOCATIONS])
    
    if st.button("Get Weather Forecast"):
        with st.spinner("Fetching weather data..."):
            forecast = get_weather_forecast(location)
        
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
        st.plotly_chart(fig, use_container_width=True)
        
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
        st.plotly_chart(fig, use_container_width=True)
        
        # Weather icons and descriptions
        st.markdown("<h4 class='sub-header'>Weather Conditions</h4>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i, row in df.iterrows():
            with cols[i % 5]:
                st.image(f"http://openweathermap.org/img/wn/{row['icon']}@2x.png", width=50)
                st.write(f"{row['date']}")
                st.write(f"{row['weather']}")
                st.write(f"{row['temp']}°C")


elif page == "Soil Distribution":
    st.markdown("<h1 class='main-header'>Soil Distribution</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Explore the soil distribution across different regions of India.
    """)
    
    # Display the soil distribution image
    soil_image_path = r"images_frontend/b.webp"  # Replace with the path to your image
    st.image(soil_image_path, caption="Soil Distribution Map of India", use_container_width=True)
    
    
    # Add a download button for the soil distribution map (optional)
    with open(soil_image_path, "rb") as file:
        btn = st.download_button(
            label="Download Soil Distribution Map",
            data=file,
            file_name="soil_distribution_map.png",
            mime="image/png"
        )




elif page == "Drone Hyperspectral Imagery":
    st.markdown("<h1 class='main-header'>Drone Hyperspectral Imagery</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Analyze vegetation health using drone hyperspectral imagery.
    Upload an image to identify healthy and stressed vegetation areas.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Analyze vegetation health
        with st.spinner("Analyzing vegetation health..."):
            # Load image
            img = cv2.imread("temp_image.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

            # Convert to HSV color space
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # Define thresholds:
            # Healthy (blue regions)
            healthy_lower = np.array([90, 50, 50])   # Lower bound for blue
            healthy_upper = np.array([140, 255, 255]) # Upper bound for blue

            # Stressed (red/yellow regions)
            stressed_lower1 = np.array([0, 100, 50])   # Lower bound for red/yellow
            stressed_upper1 = np.array([20, 255, 255]) # Upper bound for red/yellow
            stressed_lower2 = np.array([160, 100, 50]) # Second range for deep red
            stressed_upper2 = np.array([180, 255, 255]) 

            # Create masks
            healthy_mask = cv2.inRange(hsv, healthy_lower, healthy_upper)
            stressed_mask1 = cv2.inRange(hsv, stressed_lower1, stressed_upper1)
            stressed_mask2 = cv2.inRange(hsv, stressed_lower2, stressed_upper2)
            stressed_mask = cv2.bitwise_or(stressed_mask1, stressed_mask2)

            # Calculate pixel counts
            total_pixels = img.shape[0] * img.shape[1]
            healthy_pixels = np.sum(healthy_mask > 0)
            stressed_pixels = np.sum(stressed_mask > 0)

            # Compute percentage
            healthy_percentage = (healthy_pixels / total_pixels) * 100
            stressed_percentage = (stressed_pixels / total_pixels) * 100

            # Display results
            st.markdown("<h3 class='sub-header'>Vegetation Health Analysis</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style=" padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <p><strong>Healthy Vegetation:</strong> {healthy_percentage:.2f}%</p>
                <p><strong>Stressed Vegetation:</strong> {stressed_percentage:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # Visualization
            st.markdown("<h4 class='sub-header'>Visualization</h4>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(img, caption="Original Image", use_container_width=True)
            
            with col2:
                st.image(healthy_mask, caption="Healthy Areas", use_container_width=True, clamp=True)
            
            with col3:
                st.image(stressed_mask, caption="Stressed Areas", use_container_width=True, clamp=True)
# Add the chatbot button
if st.button("💬", key="chatbot_button", help="Click to open chatbot"):
    st.session_state.show_chatbot = not st.session_state.get("show_chatbot", False)

# Display the chatbot popup if the button is clicked
if st.session_state.get("show_chatbot", False):
    with st.form(key="chatbot_form"):
        st.markdown("<div class='popup-form'>", unsafe_allow_html=True)
        st.markdown("### Chatbot")
        
        # Input fields
        lang = st.selectbox("Select Language", ["English", "Hindi", "Tamil", "Telugu", "Malayalam", "Bhojpuri", "Bengali", "Manipuri", "Urdu", "Gujarati", ""
        "Kannada"])
        detail = st.text_area("Your Query", placeholder="Enter your query related to farming...")
        
        # Submit button
        if st.form_submit_button("Submit"):
            # Call the flow.py logic
            from mira_sdk import MiraClient, Flow
            from dotenv import load_dotenv
            import os

            # Load environment variables
            load_dotenv()
            api_key = os.getenv("API_KEY")

            # Initialize the client
            client = MiraClient(config={"API_KEY": api_key})
            flow = Flow(source="chatbot.yaml")

            # Prepare input dictionary
            input_dict = {"lang": lang, "detail": detail}

            # Get response from the flow
            response = client.flow.test(flow, input_dict)

            # Store the response in session state
            st.session_state.chatbot_response = response["result"]

        # Display the response if available
        if "chatbot_response" in st.session_state:
            st.markdown("### Response")
            st.write(st.session_state.chatbot_response)

        st.markdown("</div>", unsafe_allow_html=True)

# Resource Optimization Page
elif page == "Resource Optimization":
    st.markdown("<h1 class='main-header'>Resource Optimization</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Get recommendations for optimal resource conditions (Nitrogen, Phosphorus, Potassium, pH, and soil moisture) based on the crop you want to grow.
    """)
    
    # Input fields
    with st.form(key="resource_optimization_form"):
        lang = st.selectbox("Select Language", ["English", "Hindi", "Tamil", "Telugu", "Malayalam", "Bhojpuri", "Bengali", "Manipuri", "Urdu", "Gujarati", ""
        "Kannada"])
        crop = st.text_input("Enter Crop Name", placeholder="e.g., Rice, Wheat, Tomato")
        
        submitted = st.form_submit_button("Get Recommendations")
    
    if submitted:
        # Store the input data in session state or a variable
        st.session_state.user_input = {
            "lang": lang,
            "crop": crop
        }

        # Call the MiraClient and Flow logic
        from mira_sdk import MiraClient, Flow
        from dotenv import load_dotenv
        import os

        # Load environment variables
        load_dotenv()
        api_key = os.getenv("API_KEY")

        # Initialize the client
        client = MiraClient(config={"API_KEY": api_key})
        flow = Flow(source="resourceflow.yaml")

        # Prepare input dictionary
        input_dict = {
            "lang": st.session_state.user_input["lang"],
            "crop": st.session_state.user_input["crop"]
        }

        # Get response from the flow
        with st.spinner("Fetching recommendations..."):
            response = client.flow.test(flow, input_dict)
        
        # Store the response in session state
        st.session_state.resource_optimization_response = response["result"]

    # Display the response if available
    if 'resource_optimization_response' in st.session_state:
        st.markdown("<h3 class='sub-header'>Optimal Resource Recommendations</h3>", unsafe_allow_html=True)
        st.write(st.session_state.resource_optimization_response)