from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")   # you can also use yolov8s.pt for better accuracy

# Train on your dataset
results = model.train(
    data="../strap_id_dataset/data.yaml",  # dataset yaml
    epochs=50,                                   # number of epochs
    imgsz=640,                                   # image size
    batch=16,                                    # batch size (adjust to your system)
    project="models/strap_id",                   # folder to save models
    name="exp1",                                 # experiment name
    workers=4                                    # number of workers
)

# After training, best model is saved at:
# models/strap_id/exp1/weights/best.pt
