import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model for weed detection
model = YOLO("C:\\Users\\megha\\PycharmProjects\\weed\\weedbest.pt")

# Define image path
image_path = "C:\\Users\\megha\\PycharmProjects\\weed\\gettyi.jpg"  # Update with actual weed image path

# Read the image
frame = cv2.imread(image_path)

# Check if the image is loaded correctly
if frame is None:
    print("Error: Could not load image.")
    exit()

# Confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.5

# Run inference on the image
results = model.predict(frame, verbose=True)  # Verbose mode to check detection output

# Check if any objects were detected
if not results or not results[0].boxes:
    print("No weeds detected.")
else:
    # Process results
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].item()  # Correct indexing for YOLOv8
            label = result.names[int(box.cls)]  # Get class label

            # Check confidence threshold
            if confidence > CONFIDENCE_THRESHOLD:
                # Extract coordinates and convert to integers
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2

                # Bounding box color (Green for detected weeds)
                color = (0, 255, 0)  # Green for weeds
                weed_text = "Detected Weed"

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Display label, confidence, and coordinates
                text = f"{label}: {confidence:.2f}"
                coord_text = f"X: {x_center}, Y: {y_center}"

                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, weed_text, (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, coord_text, (x_min, y_max + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the processed image
cv2.imshow("Weed Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

