import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os

st.set_page_config(page_title="Banana Ripeness Classifier", layout="wide")

# Map your exact GitHub filenames to the models
MODEL_PATHS = {
    "YOLO Detection": "models/yolo_detect_best.pt",
    "Custom CNN": "models/custom_cnn_model.keras",
    "EfficientNetB0": "models/efficientnetb0_model.keras",
    "MobileNetV2": "models/mobilenetv2_model.keras"
}

# Safely load YOLO model
@st.cache_resource
def load_yolo():
    path = MODEL_PATHS["YOLO Detection"]
    if os.path.exists(path):
        try:
            return YOLO(path)
        except Exception as e:
            st.error(f"Error loading YOLO: {e}")
    else:
        st.warning(f"YOLO file not found at {path}")
    return None

# Safely load the selected CNN model
@st.cache_resource
def load_cnn(model_name):
    path = MODEL_PATHS.get(model_name)
    if path and os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            st.error(f"Error loading {model_name}: {e}")
    else:
        st.warning(f"CNN file not found at {path}")
    return None

# UI Setup
st.title("🍌 Banana Ripeness & Quality Classifier")
st.markdown("Upload an image of a banana to analyze its ripeness level and detect defects.")

# Sidebar for selecting all available models
st.sidebar.header("Control Panel")
selected_cnn_name = st.sidebar.selectbox(
    "Select Classification Model:",
    ["Custom CNN", "EfficientNetB0", "MobileNetV2"]
)

# Initialize the models
yolo_model = load_yolo()
cnn_model = load_cnn(selected_cnn_name)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        # CRASH-PROOF OPENCV IMAGE PROCESSING
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("OpenCV failed to decode image. File may be corrupted.")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("Analyze Banana"):
                with st.spinner("Processing image..."):
                    
                    # YOLO Detection Phase
                    if yolo_model is not None:
                        try:
                            # Run inference
                            results = yolo_model(img_rgb, verbose=False)
                            # Render bounding boxes
                            annotated_frame = results[0].plot()
                            st.subheader("Detection Results")
                            st.image(annotated_frame, use_container_width=True)
                        except Exception as e:
                            st.error(f"Detection error: {e}")
                    
                    # CNN Classification Phase
                    if cnn_model is not None:
                        try:
                            # Resize to match standard CNN input
                            resized_img = cv2.resize(img_rgb, (224, 224))
                            normalized_img = resized_img / 255.0
                            input_tensor = np.expand_dims(normalized_img, axis=0)
                            
                            predictions = cnn_model.predict(input_tensor, verbose=0)
                            
                            st.markdown("---")
                            st.subheader(f"Classification ({selected_cnn_name})")
                            st.success(f"Raw Output Score: {float(predictions[0][0]):.4f}")
                        except Exception as e:
                            st.error(f"Classification error: {e}")
                            
    except Exception as general_error:
        st.error(f"Critical error: {general_error}")
