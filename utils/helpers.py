"""
Utility functions and helpers for the AgriSense application
"""
import streamlit as st
import cv2
import numpy as np


def set_bg_image():
    """Set background image for the application"""
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("images_frontend/c.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            opacity: 1;
        }
        .stApp::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.5);
            z-index: -1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_custom_css():
    """Apply custom CSS styles to the application"""
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
            color: #000000;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            display: inline-block;
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


def analyze_vegetation_health(image_path):
    """
    Analyze vegetation health from hyperspectral imagery
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Define thresholds
    # Healthy (blue regions)
    healthy_lower = np.array([90, 50, 50])
    healthy_upper = np.array([140, 255, 255])

    # Stressed (red/yellow regions)
    stressed_lower1 = np.array([0, 100, 50])
    stressed_upper1 = np.array([20, 255, 255])
    stressed_lower2 = np.array([160, 100, 50])
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

    return {
        "original_image": img,
        "healthy_mask": healthy_mask,
        "stressed_mask": stressed_mask,
        "healthy_percentage": healthy_percentage,
        "stressed_percentage": stressed_percentage
    }
