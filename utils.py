import os
# This line prevents a common segmentation fault when TF and PyTorch load together
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
# Suppress the annoying CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']

def get_classification_model(model_name):
    """Loads ONLY the requested classification model to save memory."""
    if model_name == "Custom CNN":
        return tf.keras.models.load_model('models/custom_cnn_model.keras') # Removed ../
    elif model_name == "MobileNetV2":
        return tf.keras.models.load_model('models/mobilenetv2_model.keras')
    elif model_name == "EfficientNetB0":
        return tf.keras.models.load_model('models/efficientnetb0_model.keras')
    elif model_name == "YOLOv8 Classify":
        return YOLO('models/yolo_cls_best.pt')

def get_detection_model():
    """Loads the YOLO detection model."""
    return YOLO('models/yolo_detect_best.pt')

def preprocess_keras_image(image, target_size=(416, 416)):
    """Resizes and normalizes an image for Keras models."""
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def predict_classification(image, model, model_name):
    """Predicts the ripeness class using the selected model."""
    if "YOLO" in model_name:
        results = model(image)
        top1_index = results[0].probs.top1
        confidence = results[0].probs.top1conf.item()
        class_name = results[0].names[top1_index] 
        return class_name.capitalize(), confidence
    else:
        processed_img = preprocess_keras_image(image)
        predictions = model.predict(processed_img, verbose=0)
        class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        return CLASS_NAMES[class_index].capitalize(), confidence

def detect_and_count(image, detection_model):
    """Uses YOLOv8 to detect bananas (COCO Class 46), draw boxes, and count."""
    img_array = np.array(image)
    results = detection_model.predict(source=img_array, classes=[46], conf=0.25)
    count = len(results[0].boxes)
    annotated_img_bgr = results[0].plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
    
    return annotated_img_rgb, count
