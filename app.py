import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Debugging message
st.write("✅ App is running!") 

# Load trained model
@st.cache_resource
def load_model():
    st.write("🔄 Loading model...")
    try:
        model = joblib.load("model/digit_model.pkl")
        st.write("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

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

st.write("✅ Canvas created!")  # Debugging statement

def preprocess_image(image):
    """Preprocess the drawn image for model prediction."""
    try:
        image = image.convert("L")  # Convert to grayscale
        image = ImageOps.invert(image)  # Invert colors (white digit on black background)
        image = image.resize((28, 28))  # Resize to MNIST format
        image = np.array(image).astype(np.float32) / 255.0  # Normalize pixel values
        image = image.reshape(1, -1)  # Flatten for model input
        return image
    except Exception as e:
        st.error(f"❌ Error in preprocessing: {e}")
        return None

# Predict Button
if st.button("Predict Digit"):
    if canvas_result.image_data is not None:
        # Convert canvas image to PIL image
        image = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype(np.uint8))
        processed_image = preprocess_image(image)

        if processed_image is not None and model is not None:
            # Run inference
            prediction = model.predict(processed_image)[0]
            st.subheader(f"🧠 AI Prediction: {prediction}")
        else:
            st.warning("⚠️ Could not process the image or model is not loaded!")
    else:
        st.warning("⚠️ Please draw a digit first!")
