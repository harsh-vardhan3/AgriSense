# AgriSense — AI Precision Farming Assistant

AgriSense is a Streamlit-based application that helps smallholder farmers with plant disease detection, outbreak risk prediction, weather insights, soil distribution visualization, and crop recommendations.

## Features
- Plant Disease Detection using a trained CNN model
- Outbreak Prediction System based on weather, susceptibility, city and month weightage
- Weather Insights with 5-day forecast and trends
- Soil Distribution map of India
- Crop Recommendations based on soil parameters (N, P, K, pH, moisture)

## Project Structure
```
AgriSense_new/
├─ app.py                      # Main Streamlit entry (modular router)
├─ app_pages/                  # Page modules
│  ├─ home.py
│  ├─ disease_detection.py
│  ├─ outbreak_prediction.py
│  ├─ weather_insights.py
│  ├─ soil_distribution.py
│  └─ crop_recommendations.py
├─ services/                   # Business logic/services
│  ├─ disease_detection.py
│  └─ weather_service.py
├─ config/                     # Constants and settings
│  └─ constants.py
├─ utils/                      # Helpers and CSS
│  └─ helpers.py
├─ crop_recc/                  # Crop recommendation logic
│  └─ logistic.py
├─ outbreak_prediction/        # Data and predictor
│  ├─ ScorePredictor.py
│  ├─ CityWeightage.csv
│  ├─ MonthWeightage.csv
│  ├─ ViralityScoreOfDiseases.csv
│  ├─ processed_FutureWeatherPrediction.csv
│  └─ processed_PastWeatherData.csv
├─ images_frontend/            # UI images
├─ disease_info.csv            # Disease metadata
├─ supplement_info.csv         # Supplement metadata
├─ plant_disease_model_1_latest.pt  # CNN weights
├─ requirements.txt
└─ README.md
```

## Requirements
- Python 3.10+ recommended
- See `requirements.txt`

## Setup
1. Create and activate a virtual environment (recommended):
	```bash
	python -m venv .venv
	.venv\Scripts\activate
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Place model and data files:
	- `plant_disease_model_1_latest.pt` in project root
	- `disease_info.csv`, `supplement_info.csv` in project root
	- Outbreak prediction CSVs in `outbreak_prediction/`

## Run
Start the app:
```bash
streamlit run app.py
```
The app will be available at `http://localhost:8501` (or `8502` depending on your environment).

## Configuration
Key constants and paths are defined in `config/constants.py`:
- API keys (OpenWeather)
- Location list, crop types, disease list
- File paths for models and CSVs

Update values there if needed.

## Pages Overview
- Home: Overview and quick info
- Disease Detection: Upload an image (JPG/PNG) for CNN-based disease prediction
- Outbreak Prediction System: Select month, crop, disease, city → risk score gauge
- Weather Insights: 5-day forecast table + temperature & humidity trends
- Soil Distribution: Static map with download
- Crop Recommendations: Best crop based on N, P, K, pH, moisture

## Model & Dataset
- Example plant disease model: [Google Drive](https://drive.google.com/file/d/1ieQZquso2Quik18msmVVE7xtwhr6-Adz/view?usp=sharing)
- Sample dataset: [Mendeley](https://data.mendeley.com/datasets/tywbtsjrjv/1)

## Notes
- Streamlit automatically reloads on code changes.
- The app uses CPU for inference; no GPU required.
- Ensure API keys are valid for weather data.

## License
This project is for research and demonstration. If you plan to distribute, please add an appropriate license file.

