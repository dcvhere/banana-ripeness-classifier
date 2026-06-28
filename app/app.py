import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from PIL import Image
from utils import get_classification_model, get_detection_model, predict_classification, detect_and_count

st.set_page_config(page_title="Banana Quality Assessment", layout="wide")
st.title("🍌 Advanced Banana Ripeness & Quality Detection")
st.markdown("""
Welcome to the automated banana grading system. 
**User Guidance:** For best results, use clear lighting and ensure the bananas are fully visible in the frame.
""")

# Cache the models individually based on user selection
@st.cache_resource
def load_classifier(name):
    return get_classification_model(name)

@st.cache_resource
def load_detector():
    return get_detection_model()

# Sidebar Controls
st.sidebar.header("Control Panel")
selected_model_name = st.sidebar.selectbox(
    "Select Classification Model:",
    ["Custom CNN", "MobileNetV2", "EfficientNetB0", "YOLOv8 Classify"]
)

input_method = st.sidebar.radio("Select Image Input Method:", ["File Upload", "Camera Capture"])

# Pre-load the required models for this specific run
try:
    classifier_model = load_classifier(selected_model_name)
    detector_model = load_detector()
except Exception as e:
    st.error(f"Error loading models. Please check your 'models/' directory. Details: {e}")
    st.stop()

# Handle Image Input
image = None
if input_method == "File Upload":
    uploaded_file = st.sidebar.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
elif input_method == "Camera Capture":
    camera_file = st.sidebar.camera_input("Take a picture")
    if camera_file is not None:
        image = Image.open(camera_file).convert('RGB')

# Display Interface and Run Inference
if image is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Analysis Results")
        
        with st.spinner("Analyzing Ripeness..."):
            # Classification
            pred_class, conf = predict_classification(image, classifier_model, selected_model_name)
            
            st.markdown(f"**Selected Model:** {selected_model_name}")
            st.success(f"**Predicted Quality:** {pred_class}")
            st.info(f"**Confidence Score:** {conf*100:.2f}%")
        
        with st.spinner("Detecting and Counting..."):
            # Detection & Counting
            annotated_image, count = detect_and_count(image, detector_model)
            
            st.markdown("---")
            st.subheader("Detection View")
            st.metric(label="Total Bananas Detected", value=count)
            st.image(annotated_image, use_column_width=True)
else:
    st.info("👈 Please upload an image or use the camera to begin.")
