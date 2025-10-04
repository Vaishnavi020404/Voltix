import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path

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
detection_mode = st.sidebar.radio("Detection Mode", ["Upload Image", "Use Camera"])

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()

# Try multiple possible paths for the model
possible_paths = [
    script_dir / ".." / "models" / "strap_id" / "exp_fixed11" / "weights" / "best.pt",
    Path("../models/strap_id/exp_fixed11/weights/best.pt"),
    Path("models/strap_id/exp_fixed11/weights/best.pt"),
    script_dir / "models" / "strap_id" / "exp_fixed11" / "weights" / "best.pt",
]

model_path = None
for path in possible_paths:
    abs_path = path.resolve()
    if abs_path.exists():
        model_path = str(abs_path)
        st.sidebar.success(f"✅ Model found")
        break

if model_path is None:
    st.sidebar.error("❌ Model not found. Please specify the path manually.")
    custom_path = st.sidebar.text_input(
        "Enter model path:",
        value=r"C:\Users\Sudhir Pandey\Documents\GitHub\Voltix\models\strap_id\exp_fixed11\weights\best.pt"
    )
    if custom_path and Path(custom_path).exists():
        model_path = custom_path
        st.sidebar.success(f"✅ Using custom path")
    else:
        st.error("Please provide a valid model path in the sidebar.")
        st.info(f"Script location: {script_dir}")
        st.info("Searched paths:")
        for p in possible_paths:
            st.text(f"  - {p.resolve()}")
        st.stop()

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

if detection_mode == "Use Camera":
    with col1:
        st.subheader("Camera Feed")
        camera_image = st.camera_input("Take a picture")
        
        if camera_image is not None:
            # Convert to PIL Image
            image = Image.open(camera_image)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Run YOLO detection
            with st.spinner("Detecting..."):
                results = model.predict(source=img_array, conf=confidence_threshold, verbose=False)
            
            # Annotate image
            annotated_img = results[0].plot()
            
            # Convert BGR to RGB
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Display annotated image
            st.image(annotated_img_rgb, caption="Detection Results", use_container_width=True)
    
    with col2:
        st.subheader("Detection Results")
        
        if camera_image is not None:
            if len(results[0].boxes) == 0:
                st.warning("No ID/strap detected")
            else:
                st.success(f"Found {len(results[0].boxes)} object(s)")
                
                # Display detections
                for idx, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    
                    with st.container():
                        st.markdown(f"**Detection {idx + 1}:**")
                        st.write(f"- Class: {label}")
                        st.write(f"- Confidence: {conf:.2%}")
                        st.markdown("---")
        else:
            st.info("📸 Take a picture to start detection")

else:  # Upload Image mode
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert to PIL Image
            image = Image.open(uploaded_file)
            
            # Display original image
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Run YOLO detection
            with st.spinner("Detecting..."):
                results = model.predict(source=img_array, conf=confidence_threshold, verbose=False)
            
            # Annotate image
            annotated_img = results[0].plot()
            
            # Convert BGR to RGB
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Display annotated image
            st.image(annotated_img_rgb, caption="Detection Results", use_container_width=True)
    
    with col2:
        st.subheader("Detection Results")
        
        if uploaded_file is not None:
            if len(results[0].boxes) == 0:
                st.warning("No ID/strap detected")
            else:
                st.success(f"Found {len(results[0].boxes)} object(s)")
                
                # Display detections
                for idx, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    with st.container():
                        st.markdown(f"**Detection {idx + 1}:**")
                        st.write(f"- Class: {label}")
                        st.write(f"- Confidence: {conf:.2%}")
                        st.write(f"- Box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
                        st.markdown("---")
        else:
            st.info("📤 Upload an image to start detection")

# Instructions
st.markdown("---")
st.markdown("""
### Instructions:
**Camera Mode:**
- Click the camera button to take a picture
- Detection will run automatically on the captured image
- Take another picture to detect again

**Upload Mode:**
- Click "Browse files" to upload an image
- Supports JPG, JPEG, and PNG formats
- Detection results will appear on the right

**Settings:**
- Adjust the confidence threshold in the sidebar to filter detections
- Lower values = more detections (but possibly more false positives)
- Higher values = fewer, more confident detections
""")

# Footer
st.markdown("---")
st.markdown("**Powered by YOLOv8 + Streamlit**")
