import pygame
import sys
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
import threading
import queue

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
ROAD_COLOR = (50, 50, 50)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
FONT_COLOR = (255, 255, 255)
# Traffic light timing (in seconds)
NORMAL_GREEN_TIME = 5
NORMAL_YELLOW_TIME = 2
NORMAL_RED_TIME = NORMAL_GREEN_TIME + NORMAL_YELLOW_TIME

# Emergency vehicle detection parameters
EMERGENCY_DETECTION_THRESHOLD = 0.4  # Confidence threshold for emergency vehicle detection
EMERGENCY_DETECTION_COOLDOWN = 5  # Seconds to wait before checking for emergency vehicles again

# Vehicle class for simulation
class Vehicle:
    def __init__(self, lane, is_emergency=False):
        self.lane = lane  # 1 or 2
        self.is_emergency = is_emergency
        
        # Set position based on lane
        if lane == 1:  # Horizontal lane
            self.x = -80
            self.y = SCREEN_HEIGHT // 2 - 20
            self.direction = 'right'
            self.width = 80
            self.height = 40
        else:  # Vertical lane
            self.x = SCREEN_WIDTH // 2 - 20
            self.y = -80
            self.direction = 'down'
            self.width = 40
            self.height = 80
        
        # Set speed and color
        self.speed = 5 if is_emergency else 3
        self.color = RED if is_emergency else BLUE
        self.stopped = False
        
        # Add a unique ID to each vehicle for collision detection
        self.id = id(self)
        
        # Add ambulance visual elements for emergency vehicles
        if is_emergency:
            self.has_lights = True
            self.light_state = True  # For blinking effect
            self.light_timer = 0
        else:
            self.has_lights = False
    
    def update(self, traffic_lights, vehicles):
        # Check if vehicle should stop at red light
        if self.lane == 1:  # Horizontal lane
            # Stop before the junction (at x=400) for red light
            if traffic_lights[0] == 'red' and self.x < SCREEN_WIDTH // 2 - 80 and self.x > SCREEN_WIDTH // 2 - 200 and not self.is_emergency:
                self.stopped = True
            else:
                # Check for vehicle ahead to prevent overlapping
                self.stopped = self._check_vehicle_ahead(vehicles)
        else:  # Vertical lane
            # Stop before the junction (at y=300) for red light
            if traffic_lights[1] == 'red' and self.y < SCREEN_HEIGHT // 2 - 80 and self.y > SCREEN_HEIGHT // 2 - 200 and not self.is_emergency:
                self.stopped = True
            else:
                # Check for vehicle ahead to prevent overlapping
                self.stopped = self._check_vehicle_ahead(vehicles)
        
        # Move vehicle if not stopped
        if not self.stopped:
            if self.direction == 'right':
                self.x += self.speed
            elif self.direction == 'down':
                self.y += self.speed
    
    def _check_vehicle_ahead(self, vehicles):
        """Check if there's a vehicle ahead to prevent overlapping"""
        for vehicle in vehicles:
            # Skip self
            if vehicle.id == self.id:
                continue
                
            # Only check vehicles in the same lane
            if vehicle.lane != self.lane:
                continue
                
            # Check if vehicle is ahead and close
            if self.direction == 'right':
                if (vehicle.x > self.x and 
                    vehicle.x - (self.x + self.width) < 10 and
                    self.y + self.height > vehicle.y and
                    self.y < vehicle.y + vehicle.height):
                    return True
            elif self.direction == 'down':
                if (vehicle.y > self.y and 
                    vehicle.y - (self.y + self.height) < 10 and
                    self.x + self.width > vehicle.x and
                    self.x < vehicle.x + vehicle.width):
                    return True
        
        return False
    
    def draw(self, screen):
        # Draw vehicle
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        
        # Add emergency vehicle indicator (ambulance lights)
        if self.is_emergency:
            # Update light blinking state
            self.light_timer += 1
            if self.light_timer > 10:  # Blink every 10 frames
                self.light_state = not self.light_state
                self.light_timer = 0
            
            light_color = RED if self.light_state else BLUE
            
            if self.direction == 'right':
                # Draw ambulance body (white rectangle on top)
                pygame.draw.rect(screen, WHITE, (self.x + 10, self.y - 10, self.width - 20, 15))
                # Draw emergency lights on top of vehicle
                pygame.draw.circle(screen, light_color, (self.x + 20, self.y - 5), 5)
                pygame.draw.circle(screen, (not light_color), (self.x + 60, self.y - 5), 5)
                # Draw red cross (ambulance symbol)
                pygame.draw.rect(screen, RED, (self.x + 35, self.y + 10, 10, 20))
                pygame.draw.rect(screen, RED, (self.x + 25, self.y + 15, 30, 10))
            else:
                # Draw ambulance body (white rectangle on top)
                pygame.draw.rect(screen, WHITE, (self.x - 10, self.y + 10, 15, self.height - 20))
                # Draw emergency lights on top of vehicle
                pygame.draw.circle(screen, light_color, (self.x - 5, self.y + 20), 5)
                pygame.draw.circle(screen, (not light_color), (self.x - 5, self.y + 60), 5)
                # Draw red cross (ambulance symbol)
                pygame.draw.rect(screen, RED, (self.x + 10, self.y + 35, 20, 10))
                pygame.draw.rect(screen, RED, (self.x + 15, self.y + 25, 10, 30))
    
    def is_out_of_bounds(self):
        if self.direction == 'right':
            return self.x > SCREEN_WIDTH
        else:
            return self.y > SCREEN_HEIGHT

# Traffic Light class
class TrafficLight:
    def __init__(self, lane):
        self.lane = lane  # 1 or 2
        self.state = 'red' if lane == 2 else 'green'  # Lane 1 starts with green, Lane 2 with red
        self.timer = NORMAL_GREEN_TIME if lane == 1 else NORMAL_RED_TIME
        self.auto_cycle = False # Flag to control whether lights auto-cycle
        
        # Set position based on lane
        if lane == 1:  # Horizontal lane
            self.x = 320
            self.y = SCREEN_HEIGHT // 2 - 80
        else:  # Vertical lane
            self.x = SCREEN_WIDTH // 2 + 40
            self.y = 320
    
    def update(self, dt, emergency_override=None):
        # If there's an emergency override, set the state accordingly
        if emergency_override is not None:
            if emergency_override == self.lane:
                self.state = 'green'
                self.timer = NORMAL_GREEN_TIME
                self.auto_cycle = False  # Disable auto-cycling during emergency
            else:
                self.state = 'red'
                self.timer = NORMAL_RED_TIME
                self.auto_cycle = False  # Disable auto-cycling during emergency
            return
        
        # If auto-cycle is disabled and no emergency, re-enable it
        if not self.auto_cycle and emergency_override is None:
            self.auto_cycle = True
            self.timer = NORMAL_GREEN_TIME if self.state == 'green' else NORMAL_RED_TIME
        
        # Only cycle lights if auto_cycle is enabled
        if self.auto_cycle:
            # Normal traffic light cycle
            self.timer -= dt
            
            if self.timer <= 0:
                if self.state == 'green':
                    self.state = 'yellow'
                    self.timer = NORMAL_YELLOW_TIME
                elif self.state == 'yellow':
                    self.state = 'red'
                    self.timer = NORMAL_RED_TIME
                elif self.state == 'red':
                    self.state = 'green'
                    self.timer = NORMAL_GREEN_TIME
    
    def draw(self, screen):
        # Draw traffic light housing
        pygame.draw.rect(screen, (70, 70, 70), (self.x, self.y, 30, 90))
        
        # Draw lights
        red_color = RED if self.state == 'red' else (100, 0, 0)
        yellow_color = YELLOW if self.state == 'yellow' else (100, 100, 0)
        green_color = GREEN if self.state == 'green' else (0, 100, 0)
        
        pygame.draw.circle(screen, red_color, (self.x + 15, self.y + 15), 10)
        pygame.draw.circle(screen, yellow_color, (self.x + 15, self.y + 45), 10)
        pygame.draw.circle(screen, green_color, (self.x + 15, self.y + 75), 10)

# Function to run emergency vehicle detection in a separate thread
def emergency_detection_thread(model, detection_queue):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        detection_queue.put(None)  # Signal that detection thread is ending
        return
    
    last_emergency_time = 0
    cooldown = False
    
    while True:
        # Check if we should exit
        if detection_queue.qsize() > 0 and detection_queue.queue[0] == "EXIT":
            break
        
        # Check if we're in cooldown
        current_time = time.time()
        if cooldown and current_time - last_emergency_time < EMERGENCY_DETECTION_COOLDOWN:
            time.sleep(0.1)  # Short sleep to prevent CPU hogging
            continue
        
        cooldown = False
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Run detection
        results = model(frame, conf=EMERGENCY_DETECTION_THRESHOLD, verbose=False)
        
        emergency_detected = False
        emergency_lane = None
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                if class_name == 'emergency':
                    emergency_detected = True
                    
                    # Always set emergency lane to 2 (vertical lane) for ambulance
                    emergency_lane = 2
                    
                    # Put detection result in queue
                    detection_queue.put((True, emergency_lane))
                    
                    # Set cooldown
                    last_emergency_time = current_time
                    cooldown = True
                    break
            
            if emergency_detected:
                break
        
        # If no emergency vehicle detected
        if not emergency_detected and not cooldown:
            detection_queue.put((False, None))
        
        # Short sleep to prevent CPU hogging
        time.sleep(0.1)
    
    # Release resources
    cap.release()

# Main function
def main():
    # Load YOLOv8 model
    model_path = os.path.join('runs', 'detect', 'train6', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please make sure you've trained the model or update the path.")
        return
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print(f"Model loaded successfully. Classes: {model.names}")
    print("IMPORTANT: The simulation will only switch traffic signals when your trained model detects an emergency vehicle.")
    print("Show emergency vehicle images to your webcam to test the system.")
    print("When an emergency vehicle is detected, an ambulance will spawn in lane 2 and the traffic light will turn green.")
    
    # Set up detection queue and thread
    detection_queue = queue.Queue()
    detection_thread = threading.Thread(target=emergency_detection_thread, args=(model, detection_queue))
    detection_thread.daemon = True
    detection_thread.start()
    
    # Set up pygame window
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Emergency Vehicle Traffic Management Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)
    
    # Initialize traffic lights
    traffic_lights = [TrafficLight(1), TrafficLight(2)]
    
    # Initialize vehicles
    vehicles = []
    
    # Timing variables
    last_vehicle_spawn = time.time()
    vehicle_spawn_interval = 3  # seconds
    last_time = time.time()
    
    # Emergency detection variables
    emergency_override = None
    emergency_override_time = 0
    
    # Main game loop
    running = True
    while running:
        # Calculate delta time
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    # Manually spawn regular vehicle in lane 1
                    vehicles.append(Vehicle(1, False))
                elif event.key == pygame.K_2:
                    # Manually spawn regular vehicle in lane 2
                    vehicles.append(Vehicle(2, False))
        
        # Check detection queue for emergency vehicles
        while not detection_queue.empty():
            detection = detection_queue.get()
            if detection is None:
                # Detection thread has ended
                running = False
                break
            
            emergency_detected, lane = detection
            if emergency_detected:
                print(f"Emergency vehicle detected! Spawning ambulance in lane 2")
                emergency_override = lane
                emergency_override_time = current_time
                
                # Spawn an ambulance in lane 2 (vertical lane)
                # Check if there's space to spawn
                can_spawn = True
                for vehicle in vehicles:
                    if vehicle.lane == 2 and vehicle.y < 100:
                        can_spawn = False
                        break
                
                if can_spawn:
                    vehicles.append(Vehicle(2, True))  # Spawn ambulance in lane 2
        
        # Clear emergency override after a certain time
        if emergency_override is not None and current_time - emergency_override_time > NORMAL_GREEN_TIME:
            emergency_override = None
        
        # Spawn vehicles randomly
        if current_time - last_vehicle_spawn > vehicle_spawn_interval:
            # 20% chance of spawning a vehicle
            if np.random.random() < 0.2:
                lane = np.random.choice([1, 2])
                
                # Check if there's space to spawn a new vehicle
                can_spawn = True
                for vehicle in vehicles:
                    if vehicle.lane == lane:
                        if lane == 1 and vehicle.x < 100:  # For horizontal lane
                            can_spawn = False
                            break
                        elif lane == 2 and vehicle.y < 100:  # For vertical lane
                            can_spawn = False
                            break
                
                if can_spawn:
                    # Only spawn regular vehicles, no emergency vehicles in auto-spawn
                    vehicles.append(Vehicle(lane, False))
                    last_vehicle_spawn = current_time
        
        # Update traffic lights
        for light in traffic_lights:
            light.update(dt, emergency_override)
        
        # Get current traffic light states
        traffic_light_states = [light.state for light in traffic_lights]
        
        # Update vehicles
        for vehicle in vehicles[:]:
            vehicle.update(traffic_light_states, vehicles)
            if vehicle.is_out_of_bounds():
                vehicles.remove(vehicle)
        
        # Draw everything
        screen.fill((0, 100, 0))  # Green background for grass
        
        # Draw roads
        pygame.draw.rect(screen, ROAD_COLOR, (0, SCREEN_HEIGHT // 2 - 40, SCREEN_WIDTH, 80))  # Horizontal road
        pygame.draw.rect(screen, ROAD_COLOR, (SCREEN_WIDTH // 2 - 40, 0, 80, SCREEN_HEIGHT))  # Vertical road
        
        # Draw road markings
        for i in range(0, SCREEN_WIDTH, 40):
            pygame.draw.rect(screen, YELLOW, (i, SCREEN_HEIGHT // 2 - 2, 20, 4))  # Horizontal road markings
        
        for i in range(0, SCREEN_HEIGHT, 40):
            pygame.draw.rect(screen, YELLOW, (SCREEN_WIDTH // 2 - 2, i, 4, 20))  # Vertical road markings
        
        # Draw intersection
        pygame.draw.rect(screen, ROAD_COLOR, (SCREEN_WIDTH // 2 - 40, SCREEN_HEIGHT // 2 - 40, 80, 80))
        
        # Draw vehicles
        for vehicle in vehicles:
            vehicle.draw(screen)
        
        # Draw traffic lights
        for light in traffic_lights:
            light.draw(screen)
        
        # Draw status information
        status_text = f"Lane 1: {traffic_light_states[0].upper()}, Lane 2: {traffic_light_states[1].upper()}"
        if emergency_override is not None:
            status_text += f" - EMERGENCY OVERRIDE: Lane {emergency_override}"
        
        status_surface = font.render(status_text, True, FONT_COLOR)
        screen.blit(status_surface, (10, 10))
        
        # Draw instructions
        instructions = [
            "Controls:",
            "1: Spawn vehicle in Lane 1",
            "2: Spawn vehicle in Lane 2",
            "ESC: Exit simulation"
        ]
        
        for i, instruction in enumerate(instructions):
            instruction_surface = font.render(instruction, True, FONT_COLOR)
            screen.blit(instruction_surface, (10, 50 + i * 30))
        
        # Update display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(60)
    
    # Clean up
    detection_queue.put("EXIT")  # Signal detection thread to exit
    pygame.quit()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    main()
