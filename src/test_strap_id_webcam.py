from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
# The path must go up one level (..), then into models\strap_id\exp_fixed11
model = YOLO("../models/strap_id/exp_fixed11/weights/best.pt")  # adjust path if needed

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

# from ultralytics import YOLO
# import cv2
# # Note: No need to import matplotlib here, as we are only printing to terminal

# # Load trained YOLOv8 model
# # The path is correct for execution from the Volitx root or src folder
# model = YOLO("../models/strap_id/exp_fixed11/weights/best.pt")

# # Open webcam (0 = default cam, use 1 if external)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# if not cap.isOpened():
#     print("Error: Could not open webcam")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         continue

#     # Run YOLO detection (use verbose=False for cleaner terminal output)
#     results = model.predict(source=frame, conf=0.2, verbose=False) 

#     # --- Print detections in terminal (This is the required output) ---
#     if len(results[0].boxes) == 0:
#         print("No ID/strap detected")
#     else:
#         for box in results[0].boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             label = model.names[cls_id]
#             print(f"{label} detected with {conf:.2f} confidence")

#     # --- CRITICAL: Break the loop manually with Ctrl+C ---
#     # We cannot use cv2.waitKey()
    
# cap.release()
# # cv2.destroyAllWindows() # This line is also disabled
