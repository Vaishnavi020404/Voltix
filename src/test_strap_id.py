from ultralytics import YOLO

# Load trained model
model = YOLO("../models/strap_id/exp14/weights/best.pt")

# Run prediction on a single image
results = model.predict(
    source="../strap_id_dataset/test/images/img9_jpg.rf.14c0ba66547f6af5704f0d6bfbc33dec.jpg",  # change to your test image
    show=True,      # show window with detection
    conf=0.5        # confidence threshold
)

# Save results automatically in runs/detect/predict
