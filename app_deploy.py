import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Set the path to your best model (relative to the Volitx root)
MODEL_PATH = "src/models/strap_id/exp_fixed11/weights/best.pt"

# --- 1. Load the Model ---
@st.cache_resource
def load_model():
    """Loads the YOLO model only once."""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}. Check path/weights. Error: {e}")
        return None

# --- 2. Video Transformer Class for Webcam ---
# This class runs the detection on every frame from the webcam.
class YOLOTransformer(VideoTransformerBase):
    def __init__(self, model, confidence):
        self.model = model
        self.confidence = confidence

    def transform(self, frame: np.ndarray) -> np.ndarray:
        # Convert frame from RGB (Streamlit default) to BGR (YOLO plot output default)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = self.model.predict(frame_bgr, conf=self.confidence, verbose=False)
        
        # Get the annotated frame (YOLO draws boxes/labels)
        annotated_frame = results[0].plot()
        
        # Convert back to RGB for display in Streamlit
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

# --- 3. Main App Logic ---
def main():
    st.set_page_config(page_title="ID Card Detection Demo", layout="wide")
    st.title("ID Card Detection (YOLOv8 + Streamlit)")
    
    model = load_model()
    if model is None:
        return

    # --- Sidebar and Mode Selection ---
    st.sidebar.title("Configuration")
    mode = st.sidebar.radio("Select Input Mode", ("Image Upload", "Live Webcam Feed"))
    
    st.sidebar.header("Detection Settings")
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    # --- Mode 1: Image Upload (Existing Code) ---
    if mode == "Image Upload":
        st.subheader("Upload an Image to Detect ID Cards")
        uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.sidebar.image(image, caption="Original Image", use_column_width=True)
            image_np = np.array(image)

            if st.sidebar.button("Run Detection"):
                st.subheader("Results")
                with st.spinner("Analyzing image..."):
                    results = model.predict(image_np, conf=confidence, save=False)
                    annotated_frame = results[0].plot()
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    st.image(annotated_frame_rgb, caption="Image with Detections", use_column_width=True)
                    st.success(f"Detection Complete! Found {len(results[0].boxes)} ID Cards.")

    # --- Mode 2: Live Webcam Feed ---
    elif mode == "Live Webcam Feed":
        st.subheader("Real-Time ID Card Detection")
        st.warning("Ensure your model is focused on the ID card area for best results.")

        webrtc_streamer(
            key="yolo-detection",
            video_transformer_factory=lambda: YOLOTransformer(model, confidence),
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )

if __name__ == "__main__":
    main()