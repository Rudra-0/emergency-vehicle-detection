import argparse
import os
import cv2
import time
from ultralytics import YOLO
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Emergency Vehicle Detection using YOLOv8')
    parser.add_argument('--model', type=str, default='runs/detect/train6/weights/best.pt', 
                        help='Path to the trained YOLOv8 model')
    parser.add_argument('--source', type=str, default=None, 
                        help='Source for detection (image path, video path, or webcam index)')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='Confidence threshold for detections')
    parser.add_argument('--save', action='store_true', 
                        help='Save detection results')
    parser.add_argument('--output', type=str, default='results', 
                        help='Output directory for saving results')
    parser.add_argument('--view-img', action='store_true', default=True,
                        help='Display detection results')
    return parser.parse_args()

def process_image(model, image_path, conf_threshold, save_results, output_dir, view_img):
    """Process a single image for emergency vehicle detection"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Run inference
    results = model(img, conf=conf_threshold)
    
    # Process results
    annotated_img = img.copy()
    
    # Get detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # Set color based on class (emergency: red, not_emergency: blue, 16: green)
            if class_name == 'emergency':
                color = (0, 0, 255)  # Red for emergency
            elif class_name == 'not_emergency':
                color = (255, 0, 0)  # Blue for not_emergency
            else:
                color = (0, 255, 0)  # Green for class 16
            
            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f"{class_name} {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_img, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1), color, -1)
            cv2.putText(annotated_img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display results
    if view_img:
        cv2.imshow("Emergency Vehicle Detection", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save results
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, annotated_img)
        print(f"Results saved to {output_path}")
    
    return annotated_img

def process_video(model, video_path, conf_threshold, save_results, output_dir, view_img):
    """Process a video for emergency vehicle detection"""
    # Open video file or webcam
    if video_path.isdigit():
        # If source is a digit, treat as webcam index
        cap = cv2.VideoCapture(int(video_path))
        video_name = f"webcam_{video_path}"
    else:
        # Otherwise treat as video file path
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if saving results
    writer = None
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_{video_name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frames
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold)
        
        # Process results
        annotated_frame = frame.copy()
        
        # Get detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # Set color based on class (emergency: red, not_emergency: blue, 16: green)
                if class_name == 'emergency':
                    color = (0, 0, 255)  # Red for emergency
                elif class_name == 'not_emergency':
                    color = (255, 0, 0)  # Blue for not_emergency
                else:
                    color = (0, 255, 0)  # Green for class 16
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label with confidence
                label = f"{class_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps_text = f"FPS: {frame_count / elapsed_time:.2f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display results
        if view_img:
            cv2.imshow("Emergency Vehicle Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Save results
        if save_results and writer is not None:
            writer.write(annotated_frame)
    
    # Release resources
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete. Processed {frame_count} frames in {elapsed_time:.2f} seconds.")
    if save_results:
        print(f"Results saved to {output_path}")

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    print(f"Model loaded successfully. Classes: {model.names}")
    
    # Determine source type
    if args.source is None:
        # If no source is provided, use a test image from the dataset
        test_dir = os.path.join('dataset', 'test', 'images')
        if os.path.exists(test_dir) and os.listdir(test_dir):
            args.source = os.path.join(test_dir, os.listdir(test_dir)[0])
            print(f"No source provided. Using test image: {args.source}")
        else:
            print("No source provided and no test images found. Please provide a source.")
            return
    
    # Process source based on type
    if os.path.isfile(args.source) and args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        # Process image
        print(f"Processing image: {args.source}")
        process_image(model, args.source, args.conf, args.save, args.output, args.view_img)
    elif os.path.isfile(args.source) and args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process video file
        print(f"Processing video: {args.source}")
        process_video(model, args.source, args.conf, args.save, args.output, args.view_img)
    elif args.source.isdigit():
        # Process webcam
        print(f"Processing webcam: {args.source}")
        process_video(model, args.source, args.conf, args.save, args.output, args.view_img)
    else:
        print(f"Unsupported source: {args.source}")

if __name__ == "__main__":
    main()
