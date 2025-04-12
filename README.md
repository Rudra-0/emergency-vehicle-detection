# Emergency Vehicle Detection and Traffic Management System

DSATM'25 Hackathon Project

## Overview

This project implements a real-time emergency vehicle detection system using YOLOv8. The system can detect emergency vehicles and manage traffic signals to prioritize their movement. It includes both a detection module and an interactive traffic simulation to demonstrate the concept.

## Features

- **Real-time Detection**: Detect emergency vehicles in images, videos, or webcam feeds
- **Traffic Simulation**: Interactive simulation showing how traffic signals can be automatically adjusted for emergency vehicles
- **Multiple Vehicle Classification**:
  - Emergency vehicles (ambulances, fire trucks, police cars)
  - Non-emergency vehicles

## Project Structure

- `detect_emergency_vehicles.py`: Main detection script for processing images and videos
- `emergency_traffic_simulation.py`: Interactive traffic simulation with emergency vehicle detection
- `dataset/`: Training and testing data for the model
- `runs/`: Directory containing trained models

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Pygame (for simulation)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/emergency-vehicle-detection.git
cd emergency-vehicle-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Detection Script

```bash
python detect_emergency_vehicles.py --source [image_path/video_path/0 for webcam]
```

### Running the Traffic Simulation

```bash
python emergency_traffic_simulation.py
```

## Model Training

The model was trained on a custom dataset of emergency and non-emergency vehicles using YOLOv8.

```bash
python train_yolov8_fixed.py
```

## License

[MIT License](LICENSE)
