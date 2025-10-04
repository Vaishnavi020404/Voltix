import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from ultralytics import YOLO
import av
import time
from pathlib import Path
import threading

# Page configuration
st.set_page_config(
    page_title="ID Strap Detector",
    page_icon="",
    layout="wide"
)

st.title("ID Strap Detector")
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
        st.sidebar.success(f"Model loaded successfully")
        break

if model_path is None:
    st.sidebar.error("❌ Model not found. Please specify the path manually.")
    custom_path = st.sidebar.text_input(
        "Enter full model path:",
        value=r"C:\Users\Sudhir Pandey\Documents\GitHub\Voltix\models\strap_id\exp_fixed11\weights\best.pt"
    )
    if custom_path and Path(custom_path).exists():
        model_path = custom_path
        st.sidebar.success(f"Model loaded successfully")
    else:
        st.error("⚠️ Please provide a valid model path in the sidebar.")
        st.info(f"**Script location:** {script_dir}")
        st.info("**Searched paths:**")
        for p in possible_paths:
            st.code(str(p.resolve()))
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

# Shared state for detections
class DetectionState:
    def __init__(self):
        self.detections = []
        self.lock = threading.Lock()
        self.frame_count = 0
        
    def add_detection(self, detection_info):
        with self.lock:
            self.detections.append(detection_info)
            if len(self.detections) > 20:
                self.detections.pop(0)
    
    def get_detections(self):
        with self.lock:
            return self.detections.copy()
    
    def increment_frame(self):
        with self.lock:
            self.frame_count += 1
            return self.frame_count

detection_state = DetectionState()

class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.conf_threshold = confidence_threshold
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Run YOLO detection
        results = model.predict(source=img, conf=self.conf_threshold, verbose=False)
        
        # Get frame count
        frame_num = detection_state.increment_frame()
        
        # Process detections
        detections = []
        if len(results[0].boxes) == 0:
            detections.append("No ID/strap detected")
        else:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                detections.append(f"{label} ({conf:.2%})")
        
        # Add to shared state
        timestamp = time.strftime("%H:%M:%S")
        detection_state.add_detection({
            'time': timestamp,
            'frame': frame_num,
            'detections': detections
        })
        
        # Annotate frame
        annotated_img = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Webcam Feed")
    
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="yolo-object-detection",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Detection Log")
    
    # Placeholder for detection log
    detection_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Update detection log continuously
    if webrtc_ctx.state.playing:
        while webrtc_ctx.state.playing:
            detections = detection_state.get_detections()
            
            if detections:
                # Display stats
                latest = detections[-1]
                with stats_placeholder.container():
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Frames", latest['frame'])
                    with col_b:
                        num_objects = len([d for d in latest['detections'] if d != "No ID/strap detected"])
                        st.metric("Objects", num_objects)
                
                # Display detection log
                log_text = []
                for det in reversed(detections[-15:]):  # Show last 15
                    log_entry = f"[{det['time']}] Frame {det['frame']}"
                    for d in det['detections']:
                        log_entry += f"\n  • {d}"
                    log_text.append(log_entry)
                
                with detection_placeholder.container():
                    st.text_area(
                        "Recent Detections",
                        "\n\n".join(log_text),
                        height=450,
                        key=f"log_{time.time()}"
                    )
            
            time.sleep(0.5)
    else:
        st.info("Click **START** to begin live detection")

# Instructions
st.markdown("---")
st.markdown("""
### Instructions:

1. **Click the START button** above to activate your webcam
2. **Allow camera access** when your browser prompts you
3. The live feed will appear with real-time detection boxes
4. Detections will be logged in the right panel with timestamps
5. **Adjust confidence threshold** in the sidebar to filter results
6. **Click STOP** to end the session

### Troubleshooting:

- **Camera not working?** Make sure your browser has camera permissions enabled
- **No video showing?** Try refreshing the page and clicking START again
- **Slow performance?** Increase the confidence threshold to reduce processing load

""")

# Footer
st.markdown("---")
st.markdown("** Powered by YOLOv8 & Streamlit WebRTC**")
