import numpy as np
import cv2
from PIL import Image, ImageOps
import streamlit as st

st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")


def preprocess_image(image):
    """Preprocess the drawn image for model prediction."""
    
    # Convert image to grayscale
    image = image.convert("L")
    
    # Invert colors (ensure white digit on black background)
    image = ImageOps.invert(image)
    
    # Resize to 28x28 (same as MNIST dataset)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to NumPy array
    image = np.array(image).astype(np.uint8)
    
    # Apply Gaussian blur to smooth edges
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Normalize pixel values to range [0, 1]
    image = image / 255.0
    
    # Flatten the image (convert to 1D array)
    image = image.flatten().reshape(1, -1)
    
    return image
