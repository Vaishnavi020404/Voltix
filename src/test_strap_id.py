from ultralytics import YOLO

# Load the NEWLY trained model
# The path is correct, assuming you ran the training script above.
# The path must go up one level (..), then into models\strap_id\exp_fixed11
model = YOLO("../models/strap_id/exp_fixed11/weights/best.pt")

# Run prediction on a single image
results = model.predict(
    source="../strap_id_dataset/test/images/img9_jpg.rf.14c0ba66547f6af5704f0d6bfbc33dec.jpg", # ADDED '..',
    show=True,
    conf=0.5 
)