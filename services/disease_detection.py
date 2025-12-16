"""
Disease detection module for plant disease identification
"""
import torch
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image
import CNN
from config.constants import DISEASE_INFO_PATH, SUPPLEMENT_INFO_PATH, MODEL_PATH


class DiseaseDetector:
    """Disease detection class for identifying plant diseases"""
    
    def __init__(self):
        """Initialize the disease detector with model and data"""
        # Load disease and supplement information
        self.disease_info = pd.read_csv(DISEASE_INFO_PATH, encoding='cp1252')
        self.supplement_info = pd.read_csv(SUPPLEMENT_INFO_PATH, encoding='cp1252')
        
        # Load the trained model
        self.model = CNN.CNN(39)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        self.model.eval()
    
    def detect_disease(self, image):
        """
        Predict disease from an image using the trained model
        
        Args:
            image: PIL Image object
            
        Returns:
            list: List of dictionaries containing disease information
        """
        # Resize and preprocess the image
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image)
        input_data = input_data.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_data)
        output = output.detach().numpy()
        index = np.argmax(output)
        
        # Get disease details
        disease_name = self.disease_info['disease_name'][index]
        description = self.disease_info['description'][index]
        treatment = self.disease_info['Possible Steps'][index]
        supplement_name = self.supplement_info['supplement name'][index]
        supplement_image_url = self.supplement_info['supplement image'][index]
        supplement_buy_link = self.supplement_info['buy link'][index]
        
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
