from ultralytics import YOLO
import os
import yaml

# Set the current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Define paths
dataset_dir = os.path.join(script_dir, 'dataset')
data_yaml_path = os.path.join(dataset_dir, 'data.yaml')

# Create a modified data.yaml with absolute paths
with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

# Update paths to absolute paths
data_config['train'] = os.path.join(dataset_dir, 'train', 'images')
data_config['val'] = os.path.join(dataset_dir, 'valid', 'images')
data_config['test'] = os.path.join(dataset_dir, 'test', 'images')

# Save the modified config to a new file
modified_yaml_path = os.path.join(script_dir, 'modified_data.yaml')
with open(modified_yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"Modified data.yaml created at: {modified_yaml_path}")

# Initialize the model - using YOLOv8n for faster training
model = YOLO('yolov8n.pt')  # Using nano model which is smaller and faster

# Train the model with optimized settings for speed
results = model.train(
    data=modified_yaml_path,
    epochs=3,              # Minimal epochs
    imgsz=416,             # Smaller image size (416 instead of 640)
    batch=4,               # Smaller batch size
    patience=2,            # Reduced patience
    save=True,
    workers=0,             # Reduce worker threads
    cache=True,            # Cache images in RAM
    device='0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
    plots=False,           # Disable plotting during training
    optimizer='SGD',       # SGD can be faster than Adam in some cases
    fraction=0.2,          # Use only a fraction of the dataset (20%)
    profile=True           # Profile the training process
)

# Evaluate the model on the validation set
val_results = model.val()

# Export the model to ONNX format for deployment
# Fixed: Remove the fraction parameter which caused the error
try:
    # Export with correct parameters for ONNX format
    model.export(format='onnx', imgsz=416)
    print("Model exported to ONNX format successfully!")
except Exception as e:
    print(f"Error exporting to ONNX: {e}")
    print("Continuing with the trained PyTorch model...")

print("Training completed successfully!")
print(f"Model saved to {os.path.join('runs', 'detect', 'train')}")

# Create a simple inference script
inference_script = '''
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
'''

# Save the inference script
with open(os.path.join(script_dir, 'inference.py'), 'w') as f:
    f.write(inference_script)

print("Inference script created at:", os.path.join(script_dir, 'inference.py'))
