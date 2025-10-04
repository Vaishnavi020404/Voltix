import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="ID + Strap Detector",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 ID + Strap Detector")
st.markdown("Real-time detection using YOLOv8")

# Sidebar controls
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.2, 0.05)
camera_index = st.sidebar.selectbox("Camera Index", [0, 1, 2], index=0)

# Model path
model_path = "../models/strap_id/exp_fixed11/weights/best.pt"

@st.cache_resource
def load_model(path):
    """Load YOLO model with caching"""
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model(model_path)

if model is None:
    st.error("Failed to load YOLO model. Please check the model path.")
    st.stop()

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    frame_placeholder = st.empty()

with col2:
    st.subheader("Detection Log")
    detection_placeholder = st.empty()
    stats_placeholder = st.empty()

# Control buttons
start_button = st.button("Start Detection", type="primary")
stop_button = st.button("Stop Detection", type="secondary")

# Session state for camera control
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False

if start_button:
    st.session_state.run_detection = True

if stop_button:
    st.session_state.run_detection = False

# Main detection loop
if st.session_state.run_detection:
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        st.error(f"Error: Could not open webcam with index {camera_index}")
        st.session_state.run_detection = False
    else:
        detection_log = []
        frame_count = 0
        
        while st.session_state.run_detection:
            ret, frame = cap.read()
            
            if not ret:
                st.warning("Failed to grab frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Run YOLO detection
            results = model.predict(source=frame, conf=confidence_threshold, verbose=False)
            
            # Annotate frame
            annotated_frame = results[0].plot()
            
            # Convert BGR to RGB for display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            # Process detections
            current_detections = []
            if len(results[0].boxes) == 0:
                current_detections.append("No ID/strap detected")
            else:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    detection_text = f"{label} detected with {conf:.2f} confidence"
                    current_detections.append(detection_text)
            
            # Update detection log (keep last 20 entries)
            timestamp = time.strftime("%H:%M:%S")
            for detection in current_detections:
                detection_log.append(f"[{timestamp}] {detection}")
            
            detection_log = detection_log[-20:]  # Keep only last 20
            
            # Display detection log
            with detection_placeholder.container():
                st.text_area(
                    "Recent Detections",
                    "\n".join(reversed(detection_log)),
                    height=300,
                    key=f"log_{frame_count}"
                )
            
            # Display stats
            with stats_placeholder.container():
                st.metric("Frames Processed", frame_count)
                st.metric("Current Detections", len(results[0].boxes))
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.03)
        
        cap.release()
        st.success("Detection stopped")
else:
    st.info("👆 Click 'Start Detection' to begin")

# Footer
st.markdown("---")
st.markdown("**Note:** Make sure your model path is correct and the webcam is accessible.")