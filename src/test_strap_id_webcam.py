from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO("../models/strap_id/exp14/weights/best.pt")  # adjust path if needed

# Open webcam (0 = default cam, use 1 if external)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    # Run YOLO detection
    results = model.predict(source=frame, conf=0.2)  # lower conf if needed

    # Draw bounding boxes + labels on frame
    annotated_frame = results[0].plot()

    # Show webcam feed
    cv2.imshow("ID + Strap Detector", annotated_frame)

    # Print detections in terminal
    if len(results[0].boxes) == 0:
        print("No ID/strap detected")
    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            print(f"{label} detected with {conf:.2f} confidence")

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
