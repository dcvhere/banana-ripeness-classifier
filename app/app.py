import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os

st.set_page_config(page_title="Advanced Banana Quality Assessment", layout="wide")

# 1. Update these to your exact 6 Roboflow category labels
CLASS_NAMES = ["Label_1", "Label_2", "Label_3", "Label_4", "Label_5", "Label_6"]

# 2. Map exact paths to the models based on the project structure
MODEL_PATHS = {
    "YOLO Detection (Counting)": "models/yolo_detect_best.pt",
    "YOLOv8 Classification": "models/yolo_cls_best.pt",
    "Custom CNN": "models/custom_cnn_model.keras",
    "EfficientNetB0": "models/efficientnetb0_model.keras",
    "MobileNetV2": "models/mobilenetv2_model.keras",
    "ResNet50": "models/resnet50_model.keras"
}

# Safely Load YOLO Detection Model
@st.cache_resource
def load_detector():
    path = MODEL_PATHS["YOLO Detection (Counting)"]
    if os.path.exists(path):
        try:
            return YOLO(path)
        except Exception as e:
            st.error(f"Error loading detector: {e}")
    return None

# Safely Load Selected Classification Model
@st.cache_resource
def load_classifier(model_name):
    path = MODEL_PATHS.get(model_name)
    if not path or not os.path.exists(path):
        return None
    try:
        if "yolo" in path.lower():
            return YOLO(path)
        else:
            return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None

# UI Setup
st.title("🍌 Banana Ripeness, Quality Detection, and Counting")
st.markdown("Upload an image or use your camera to classify ripeness and count the bananas present.")

# Sidebar Configuration
st.sidebar.header("Control Panel")
selected_classifier = st.sidebar.selectbox(
    "Select Classification Model:",
    ["Custom CNN", "EfficientNetB0", "MobileNetV2", "ResNet50", "YOLOv8 Classification"]
)

input_method = st.sidebar.radio("Choose Input Method:", ["File Upload", "Camera Capture"])

# Initialize Models
detector = load_detector()
classifier = load_classifier(selected_classifier)

# Input Handling
image_file = None
if input_method == "File Upload":
    image_file = st.sidebar.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.sidebar.camera_input("Take a picture")

# Main Execution Flow
if image_file is not None:
    try:
        # CRASH-PROOF OPENCV DECODING
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("OpenCV failed to decode the image. File may be corrupted.")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Step 1: Preview Image
        st.image(img_rgb, caption="Input Image Ready for Analysis", use_container_width=False, width=400)
        
        # Step 2: Explicit Action Button
        if st.button("Analyze Image", type="primary", use_container_width=True):
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("1. Object Detection & Counting")
                if detector:
                    with st.spinner("Detecting and counting bananas..."):
                        # YOLO Inference
                        det_results = detector(img_rgb, verbose=False)
                        boxes = det_results[0].boxes
                        banana_count = len(boxes)
                        annotated_frame = det_results[0].plot()
                        
                        st.metric(label="Total Bananas Detected", value=banana_count)
                        st.image(annotated_frame, caption="YOLOv8 Bounding Boxes", use_container_width=True)
                else:
                    st.error("YOLO detection model is missing from the models directory.")
                    
            with col2:
                st.subheader(f"2. Ripeness Classification ({selected_classifier})")
                if classifier:
                    with st.spinner("Classifying ripeness quality..."):
                        if "YOLO" in selected_classifier:
                            cls_results = classifier(img_rgb, verbose=False)
                            top_class = cls_results[0].names[cls_results[0].probs.top1]
                            confidence = cls_results[0].probs.top1conf.item() * 100
                            
                            st.success(f"**Predicted Quality:** {top_class}")
                            st.info(f"**Confidence Score:** {confidence:.2f}%")
                        else:
                            # Standard CNN / Transfer Learning Pipeline
                            # Resized to 416x416 to match the project's Roboflow dataset preprocessing
                            resized_img = cv2.resize(img_rgb, (416, 416))
                            normalized_img = resized_img / 255.0
                            input_tensor = np.expand_dims(normalized_img, axis=0)
                            
                            predictions = classifier.predict(input_tensor, verbose=0)[0]
                            class_index = np.argmax(predictions)
                            confidence = predictions[class_index] * 100
                            
                            try:
                                predicted_label = CLASS_NAMES[class_index]
                            except IndexError:
                                predicted_label = f"Class {class_index}"
                                
                            st.success(f"**Predicted Quality:** {predicted_label}")
                            st.info(f"**Confidence Score:** {confidence:.2f}%")
                else:
                    st.error(f"{selected_classifier} model is missing from the models directory.")
                    
    except Exception as e:
        st.error(f"Critical error processing image: {e}")
else:
    st.info("👈 Please select an input method and provide an image to begin.")
