from ultralytics import YOLO
import cv2
import os

# Load the trained model
model_path = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)

# Function to run inference on an image
def detect_emergency_vehicles(image_path):
    results = model(image_path)
    
    # Process and display results
    for result in results:
        # Get the original image
        img = cv2.imread(image_path)
        
        # Draw bounding boxes
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow("Emergency Vehicle Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Replace with the path to your test image
    test_image = os.path.join('dataset', 'test', 'images', os.listdir(os.path.join('dataset', 'test', 'images'))[0])
    detect_emergency_vehicles(test_image)
