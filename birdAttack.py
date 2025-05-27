import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time
from gpiozero.pins.pigpio import PiGPIOFactory 
from gpiozero import AngularServo

# Initialize camera with higher resolution
picam2 = Picamera2()
dispW, dispH = 1280, 720
picam2.preview_configuration.main.size = (dispW, dispH)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 30
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOv11 model for bird detection
# Load the YOLO11 model
modelo = YOLO("yolo11n.pt")

# Export the model to NCNN format
modelo.export(format="ncnn")  # creates '/yolo11n_ncnn_model'

# Load the exported NCNN model
model = YOLO("./yolo11n_ncnn_model")
OBJECT_CLASS = 14  # Class ID for birds in COCO dataset
MIN_CONFIDENCE = 0.8  # Minimum confidence threshold

# Tracking control variables
track = 0  # Default to training mode (0: off, 1: on)

# Display configuration
font = cv2.FONT_HERSHEY_SIMPLEX
FPS_POS = (30, 60)
FONT_SIZE = 1.5
FONT_WEIGHT = 3
TEXT_COLOR = (0, 0, 255)

def onTrack7(val):
    """Toggle between training and tracking modes"""
    global track
    track = val
    print(f'Tracking Mode: {"ON" if track else "OFF"}')

# Create control window and trackbar
cv2.namedWindow('AI Tracker')
cv2.createTrackbar('Tracking Mode', 'AI Tracker', 0, 1, onTrack7)

while True:
    loop_start = time.time()
    frame = picam2.capture_array()
    
    # Run YOLO object detection
    results = model(frame, imgsz=320, conf=MIN_CONFIDENCE)
    annotated_frame = results[0].plot()  # Get annotated frame with boxes
    
    # Process detections only in tracking mode
    if track == 1:
        best_detection = None
        max_confidence = 0

        # Extract detection information
        for detection in results[0].boxes:
            cls = int(detection.cls.item())
            conf = detection.conf.item()
            
            # Only consider bird detections
            if cls == OBJECT_CLASS and conf > max_confidence:
                max_confidence = conf
                best_detection = detection

        if best_detection is not None:
            # Get bounding box coordinates
            x1, y1, x2, y2 = best_detection.xyxy[0].tolist()
            
            # Calculate center of detection
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Draw center point
            cv2.circle(annotated_frame, (int(center_x), int(center_y)), 
                       5, (0, 255, 0), -1)

            # Calculate position errors relative to frame center
            pan_error = center_x - (dispW / 2)
            tilt_error = center_y - (dispH / 2)

    # Display performance metrics
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time if inference_time != 0 else 0
    cv2.putText(annotated_frame, f'FPS: {fps:.1f}', FPS_POS,
                font, FONT_SIZE, TEXT_COLOR, FONT_WEIGHT)
    
    # Display tracking status
    mode_text = f'Tracking: {"ACTIVE" if track else "INACTIVE"}'
    cv2.putText(annotated_frame, mode_text, (30, 120), 
                font, FONT_SIZE, TEXT_COLOR, FONT_WEIGHT)

    cv2.imshow("AI Tracker", annotated_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

    # Calculate adaptive FPS
    loop_time = time.time() - loop_start
    fps = 0.9 * fps + 0.1 * (1 / loop_time)

# Cleanup
cv2.destroyAllWindows()
