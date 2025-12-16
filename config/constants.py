"""
Configuration constants for the AgriSense application
"""

# API Keys
API_KEY = "sb-bef7c0288aabf67ffbceb0a3a99528d8"
API_KEY2 = "f82b855ceef9e8f9a7fd1ccafda58615"

# Locations in Manipur
LOCATIONS = [
    {"name": "Imphal East", "lat": 24.8060, "lon": 93.9387},
    {"name": "Imphal West", "lat": 24.8065, "lon": 93.9389},
    {"name": "Thoubal", "lat": 24.6388, "lon": 94.0033},
    {"name": "Bishnupur", "lat": 24.6276, "lon": 93.7739},
    {"name": "Churachandpur", "lat": 24.3333, "lon": 93.6833},
    {"name": "Chandel", "lat": 24.3167, "lon": 94.0333},
    {"name": "Senapati", "lat": 25.2675, "lon": 94.0667},
    {"name": "Ukhrul", "lat": 25.1167, "lon": 94.3667},
    {"name": "Tamenglong", "lat": 24.9750, "lon": 93.5167},
    {"name": "Jiribam", "lat": 24.8000, "lon": 93.1167}
]

# Crop types common in Manipur
CROP_TYPES = [
    "Apple", "Blueberry", "Cherry", "Corn", "Grape", 
    "Orange", "Peach", "Pepper", "Potato", "Raspberry", 
    "Soybean", "Squash", "Strawberry", "Tomato"
]

# Supported diseases
DISEASES = [
    "Apple Black Rot",
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
    "Tomato Spider Mite Infestation"
]

# Cities for outbreak prediction
CITIES = ["Imphal West", "Imphal East", "Churachandpur", "Thoubal", "Senapati"]

# Languages
LANGUAGES = [
    "English", "Hindi", "Tamil", "Telugu", "Malayalam", 
    "Bhojpuri", "Bengali", "Manipuri", "Urdu", "Gujarati", "Kannada"
]

# Months
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# File paths
DISEASE_INFO_PATH = 'disease_info.csv'
SUPPLEMENT_INFO_PATH = 'supplement_info.csv'
MODEL_PATH = 'plant_disease_model_1_latest.pt'
SOIL_IMAGE_PATH = r"images_frontend/b.webp"
HOME_IMAGE_PATH = r"images_frontend/a.jpg"
BACKGROUND_IMAGE_PATH = "images_frontend/c.jpg"

# Outbreak prediction file paths
OUTBREAK_PATHS = {
    "future_weather": "outbreak_prediction/processed_FutureWeatherPrediction.csv",
    "past_weather": "outbreak_prediction/processed_PastWeatherData.csv",
    "crop_susceptibility": "outbreak_prediction/DiseaseSusceptibilityOfCrops.csv",
    "disease_virality": "outbreak_prediction/ViralityScoreOfDiseases.csv",
    "city_weightage": "outbreak_prediction/CityWeightage.csv",
    "month_weightage": "outbreak_prediction/MonthWeightage.csv"
}
