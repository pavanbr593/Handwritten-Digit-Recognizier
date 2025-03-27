import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import cv2
from streamlit_drawable_canvas import st_canvas

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model/digit_model.pkl")

model = load_model()

# Streamlit UI
st.title("📝 Handwritten Digit Recognition ✨")
st.write("Draw a digit (0-9) and the AI will recognize it!")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

def preprocess_image(image):
    """Preprocess drawn image for model prediction."""
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors
    image = image.resize((28, 28))  # Resize to match MNIST dataset
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = image.flatten().reshape(1, -1)  # Flatten for model
    return image

# Predict Button
if st.button("Predict Digit"):
    if canvas_result.image_data is not None:
        # Convert drawn image to PIL
        image = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype(np.uint8))
        processed_image = preprocess_image(image)
        
        # Run inference
        prediction = model.predict(processed_image)[0]

        st.subheader(f"🧠 AI Prediction: {prediction}")
    else:
        st.warning("Please draw a digit first!")
