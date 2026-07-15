import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os

# Set page layout configuration
st.set_page_config(page_title="Banana Ripeness Classifier", layout="centered")

# Cache model loading to prevent memory leaks on Cloud Run
@st.cache_resource
def load_models():
    models = {}
    
    # Paths to your weight files - update names if they differ
    yolo_path = "models/best.pt"
    cnn_path = "models/banana_model.keras"
    
    # Safe loading for YOLO
    if os.path.exists(yolo_path):
        try:
            models['yolo'] = YOLO(yolo_path)
        except Exception as e:
            st.error(f"Error loading YOLO model: {e}")
    else:
        st.warning(f"YOLO model file not found at {yolo_path}. Object detection will be disabled.")
        models['yolo'] = None

    # Safe loading for TensorFlow CNN
    if os.path.exists(cnn_path):
        try:
            models['cnn'] = tf.keras.models.load_model(cnn_path)
        except Exception as e:
            st.error(f"Error loading CNN model: {e}")
    else:
        st.warning(f"CNN model file not found at {cnn_path}. Classification will be disabled.")
        models['cnn'] = None
        
    return models

# Initialize models safely
models = load_models()

st.title("🍌 Banana Ripeness & Quality Classifier")
st.write("Upload an image of a banana to analyze its ripeness level and detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        # --- CRASH-PROOF OPENCV IMAGE PROCESSING ---
        # 1. Read file bytes directly from Streamlit memory buffer
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        # 2. Decode bytes into a standard BGR image array using OpenCV
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("OpenCV failed to decode image. File may be corrupted.")
            
        # 3. Convert to RGB layout for display and model inference
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Display the uploaded image cleanly
        st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
        
        # Execution button
        if st.button("Analyze Banana"):
            with st.spinner("Processing image through pipeline..."):
                
                # --- PHASE 1: Object Detection (YOLOv8) ---
                if models['yolo'] is not None:
                    try:
                        # OpenCV formats fit seamlessly into YOLO
                        yolo_results = models['yolo'](img_rgb, verbose=False)
                        # Process YOLO outputs here if needed (e.g., drawing bounding boxes)
                    except Exception as e:
                        st.error(f"Object detection phase encountered an error: {e}")
                
                # --- PHASE 2: Classification (CNN) ---
                if models['cnn'] is not None:
                    try:
                        # OpenCV allows ultra-safe image resizing to match your CNN input shape
                        # Update (224, 224) to your model's required input resolution
                        resized_img = cv2.resize(img_rgb, (224, 224))
                        
                        # Normalize and add batch dimension
                        normalized_img = resized_img / 255.0
                        input_tensor = np.expand_dims(normalized_img, axis=0)
                        
                        # Inference execution
                        predictions = models['cnn'].predict(input_tensor, verbose=0)
                        
                        # Placeholder: Adjust logic based on your specific output classes
                        score = float(predictions[0][0]) if len(predictions[0]) == 1 else np.argmax(predictions[0])
                        st.success(f"Analysis complete! Classification Output Score: {score:.4f}")
                        
                    except Exception as e:
                        st.error(f"Classification phase encountered an error: {e}")
                        
    except Exception as general_error:
        st.error(f"Critical error handling the uploaded file: {general_error}")
