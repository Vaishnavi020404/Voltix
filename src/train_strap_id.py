from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt") 

# Train on your dataset
results = model.train(
    # --- CORRECTED PATH (relative to Volitx folder) ---
   # CHANGE THIS LINE in train_strap_id.py (Line 9)
    # In train_strap_id.py (Line 9)
    data="../data.yaml", # No '..' or absolute path needed
    # ---------------------------------------------------
    epochs=100,                         # Recommended epochs for good learning
    imgsz=640,
    batch=16, 
    project="models/strap_id",          
    name="exp_fixed",                   # Use a new name like this
    workers=4 
)