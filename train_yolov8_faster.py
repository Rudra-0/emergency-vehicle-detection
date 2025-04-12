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

# Initialize the model - using YOLOv8n-TINY for faster training
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
try:
    # Export with correct parameters for ONNX format
    model.export(format='onnx', imgsz=416)
    print("Model exported to ONNX format successfully!")
except Exception as e:
    print(f"Error exporting to ONNX: {e}")
    print("Continuing with the trained PyTorch model...")

print("Training completed successfully!")
print(f"Model saved to {os.path.join('runs', 'detect', 'train')}")
