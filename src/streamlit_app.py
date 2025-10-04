import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from ultralytics import YOLO
import numpy as np
import av
import time
from pathlib import Path
import queue
import threading

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

# Detection log queue
detection_queue = queue.Queue(maxsize=20)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.confidence = confidence_threshold
        self.frame_count = 0
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Run YOLO detection
        results = self.model.predict(source=img, conf=self.confidence, verbose=False)
        
        # Annotate frame
        annotated_img = results[0].plot()
        
        # Process detections for log
        detections = []
        if len(results[0].boxes) == 0:
            detections.append("No ID/strap detected")
        else:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]
                detections.append(f"{label}: {conf:.2f}")
        
        # Add to queue
        timestamp = time.strftime("%H:%M:%S")
        try:
            detection_queue.put_nowait({
                'time': timestamp,
                'detections': detections,
                'count': len(results[0].boxes)
            })
        except queue.Full:
            try:
                detection_queue.get_nowait()
                detection_queue.put_nowait({
                    'time': timestamp,
                    'detections': detections,
                    'count': len(results[0].boxes)
                })
            except:
                pass
        
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    webrtc_ctx = webrtc_streamer(
        key="yolo-detector",
        video_transformer_factory=VideoTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Detection Log")
    detection_log_placeholder = st.empty()
    
    # Display detection log
    if webrtc_ctx.state.playing:
        detection_log = []
        while True:
            try:
                if webrtc_ctx.state.playing:
                    try:
                        data = detection_queue.get(timeout=1)
                        log_entry = f"[{data['time']}] {', '.join(data['detections'])}"
                        detection_log.append(log_entry)
                        detection_log = detection_log[-15:]  # Keep last 15
                        
                        detection_log_placeholder.text_area(
                            "Recent Detections",
                            "\n".join(reversed(detection_log)),
                            height=400
                        )
                    except queue.Empty:
                        pass
                else:
                    break
                    
                time.sleep(0.1)
            except:
                break
    else:
        st.info("👈 Click 'START' to begin detection")

# Instructions
st.markdown("---")
st.markdown("""
### Instructions:
1. Click **START** button above to activate your webcam
2. Allow browser access to your camera when prompted
3. The model will detect IDs and straps in real-time
4. Adjust confidence threshold in the sidebar if needed
5. Click **STOP** to end the session

**Note:** Make sure your browser has camera permissions enabled.
""")

# Footer
st.markdown("---")
st.markdown("**Powered by YOLOv8 + Streamlit WebRTC**")
