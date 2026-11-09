"""
System Configuration Settings for AgriVision AI Plant Leaf Detection & Smart Rover.
Contains hardware pinouts, detection thresholds, motor parameters, water pump safety limits,
and automatic image storage configuration.
"""

import os
import platform

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------------
# Platform Detection (Laptop vs Raspberry Pi)
# -------------------------------------------------------------
IS_RASPBERRY_PI = False
try:
    if os.path.exists('/sys/firmware/devicetree/base/model'):
        with open('/sys/firmware/devicetree/base/model', 'r') as f:
            if 'Raspberry Pi' in f.read():
                IS_RASPBERRY_PI = True
    elif platform.machine().startswith(('arm', 'aarch')):
        IS_RASPBERRY_PI = True
except Exception:
    IS_RASPBERRY_PI = False

# -------------------------------------------------------------
# Camera & Video Feed Settings
# -------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
STREAM_FPS = 15

# -------------------------------------------------------------
# Object Detection & AI Parameters
# -------------------------------------------------------------
OBJECT_DETECTION_ENABLED = True
DISEASE_DETECTION_ENABLED = True
DEFAULT_CONFIDENCE_THRESHOLD = 70.0  # Percentage
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")

# Known obstacle classes in COCO that should stop the rover
OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "cat", "dog", "horse", "sheep", "cow", "chair", "couch",
    "bed", "dining table", "toilet", "tv", "laptop", "bottle"
}

# -------------------------------------------------------------
# Motor Navigation Settings
# -------------------------------------------------------------
SPEED_LOW = 35
SPEED_MEDIUM = 65
SPEED_HIGH = 100
DEFAULT_SPEED = SPEED_MEDIUM

MOTOR_LEFT_ENA = 12   # Left motor PWM speed pin (BCM)
MOTOR_LEFT_IN1 = 17   # Left motor direction 1
MOTOR_LEFT_IN2 = 27   # Left motor direction 2

MOTOR_RIGHT_ENB = 13  # Right motor PWM speed pin (BCM)
MOTOR_RIGHT_IN3 = 22  # Right motor direction 1
MOTOR_RIGHT_IN4 = 23  # Right motor direction 2

# Ultrasonic Obstacle Sensor (optional hardware)
ULTRASONIC_TRIG_PIN = 20
ULTRASONIC_ECHO_PIN = 21

# -------------------------------------------------------------
# Water Pump & Sprayer Settings (WATER ONLY)
# -------------------------------------------------------------
WATER_PUMP_ENABLED = True
PUMP_RELAY_PIN = 24    # Sprayer Relay Pin (BCM)
STATUS_LED_PIN = 25    # Status LED

DEFAULT_SPRAY_DURATION = 2.0    # Seconds to spray
DEFAULT_SPRAY_INTERVAL = 4.0    # Seconds minimum cooldown between sprays
MAX_PUMP_RUNTIME = 6.0          # Max continuous runtime safety cutoff
AUTO_SPRAY_CONFIDENCE_MIN = 70.0

# -------------------------------------------------------------
# Automatic Image Storage Settings
# -------------------------------------------------------------
AUTO_STORAGE_ENABLED = True
DETECTED_IMAGES_DIR = os.path.join(BASE_DIR, "detected_images")
DETECTION_LOGS_DIR = os.path.join(BASE_DIR, "detection_logs")
CSV_LOG_PATH = os.path.join(DETECTION_LOGS_DIR, "detection_history.csv")
SYSTEM_LOG_PATH = os.path.join(DETECTION_LOGS_DIR, "system_logs.txt")

SAVE_HEALTHY_IMAGES = True
SAVE_DISEASED_IMAGES = True
MIN_IMAGE_CAPTURE_INTERVAL = 3.5  # Min seconds between saving duplicate images of same category
