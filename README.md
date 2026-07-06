# 🍌 Advanced Banana Ripeness & Quality Detection

An end-to-end computer vision application designed to automatically grade the ripeness of bananas and count inventory in real-time. 

🚀 **[Experience the Live Web App on Hugging Face] (https://banana-ripeness-classifier-dcvhere.streamlit.app/)](https://banana-ripeness-classifier-dcvhere.streamlit.app/)**

---

## 📸 System in Action

*(Replace the placeholder links below with actual screenshots of your app running. You can upload images to your repository and link them here).*

![Classification Preview](l[ink_to_your_classification_screenshot.png](https://github.com/dcvhere/banana-ripeness-classifier/blob/main/images/Screenshot%20From%202026-06-28%2022-03-18.png))
*Real-time ripeness classification using deep learning.*

![Detection Preview](link_to_your_detection_screenshot.png)
*Instance detection and counting via YOLOv8.*

---

## 🧠 Methodology & Architecture

This project implements a dual-model approach, separating the classification logic from the object detection logic to optimize inference speed and accuracy. 

### 1. Ripeness Classification (Quality Assurance)
The system evaluates individual bananas and classifies them into one of four categories: `Unripe`, `Ripe`, `Overripe`, or `Rotten`. Users can select between multiple trained architectures to compare performance:
* **Custom CNN:** A baseline convolutional neural network built from scratch.
* **MobileNetV2:** A lightweight, highly efficient transfer-learning model optimized for edge devices.
* **EfficientNetB0:** A scaled architecture balancing high accuracy with computational efficiency.
* **YOLOv8 (Classify):** State-of-the-art classification utilizing the Ultralytics framework.

### 2. Inventory Counting (Object Detection)
* **YOLOv8 (Detect):** A custom-trained YOLOv8 detection model specifically tuned to identify and draw bounding boxes around bananas (COCO Class 46) to provide automated inventory counts regardless of the background environment.

### 3. Deployment
The web interface is built using **Streamlit** and deployed inside a containerized **Docker** environment on Hugging Face Spaces. It utilizes `opencv-python-headless` to ensure stable, crash-free image processing on headless Linux servers.

---

## 💻 Local Setup & Installation

If you would like to run this application locally on your own machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/dcvhere/banana-ripeness-classifier.git](https://github.com/dcvhere/banana-ripeness-classifier.git)
cd banana-ripeness-classifier
