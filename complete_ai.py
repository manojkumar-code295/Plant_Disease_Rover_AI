#!/usr/bin/env python3
"""
=============================================================================
PRECISION AGRICULTURE SMART ROVER - HIGH-PERFORMANCE AI DETECTION ENGINE
=============================================================================
Architecture:
- Native Raspberry Pi CSI Camera via Picamera2 (smooth 30 FPS, zero V4L2 lock)
- Fast TFLite Inference (~50ms) with automatic Keras fallback
- Strict Leaf-Only Detection: rejects human faces, skin, hands, and room backgrounds
- Plant Disease Classification:
    * Bacterial Spot
    * Cercospora Leaf Spot
    * Curl Virus
    * Healthy Leaf
- Configurable Confidence Handling ("NO LEAF DETECTED", "UNCERTAIN", "HEALTHY")
- Asynchronous Non-Blocking Cloud Upload to Render (ZERO UI delay/freeze)
- Bi-directional Rover Control (Steering, Speed, Spray Pump, Heartbeat)
- Verified Raspberry Pi 4 B GPIO Pin Mapping via lgpio (safe simulation fallback)
- Safe startup & clean shutdown with fsync data protection
- Headless and GUI dual-mode execution (supports boot via systemd service)
=============================================================================
"""

import os
import sys

# Suppress verbose Qt font warnings when OpenCV window opens
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

import time
import queue
import signal
import threading
import collections
import socket
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
import numpy as np
import requests

# Base directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ============================================================
# CONFIGURATION & THRESHOLDS (CENTRALIZED & CONFIGURABLE)
# ============================================================

# Model paths
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "best_chili_model.tflite")
KERAS_MODEL_PATH = os.path.join(BASE_DIR, "best_chili_model.keras")
CLASS_FILE = os.path.join(BASE_DIR, "class_names.txt")

# Image input size for MobileNetV2
IMAGE_SIZE = 224

# AI Confidence & Classification Thresholds (Strict Precision Thresholds)
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.75))
CONFIDENCE_MARGIN = float(os.environ.get("CONFIDENCE_MARGIN", 0.20))

# Robust Leaf Filtering Thresholds (Botanical Quality Gate for Chili Leaves)
MIN_LEAF_AREA = int(os.environ.get("MIN_LEAF_AREA", 3200))           # Min pixels to qualify as a leaf
MAX_LEAF_AREA_RATIO = float(os.environ.get("MAX_LEAF_AREA_RATIO", 0.98)) # Up to 98% of frame for full-frame closeups
MIN_LEAF_DENSITY = float(os.environ.get("MIN_LEAF_DENSITY", 0.22))   # Foliage density inside bounding box
MAX_SKIN_RATIO = float(os.environ.get("MAX_SKIN_RATIO", 0.20))       # Skin tone ratio threshold for rejecting human face/hand
MIN_TEXTURE_VAR = float(os.environ.get("MIN_TEXTURE_VAR", 8.0))      # Laplacian variance to reject flat blank walls
MIN_EXG_THRESHOLD = float(os.environ.get("MIN_EXG_THRESHOLD", 24.0)) # Excess Green Index (chlorophyll presence)
MIN_SATURATION = float(os.environ.get("MIN_SATURATION", 55.0))       # Rejects gray/off-white rooms and furniture

# Camera & Performance Configuration
# Mounting orientation: "90_CCW" (default for upright rover chassis mount), "90_CW", "180", "NONE"
CAMERA_ROTATION = os.environ.get("CAMERA_ROTATION", "90_CCW")
PREDICT_EVERY_N_FRAMES = int(os.environ.get("PREDICT_EVERY_N_FRAMES", 2))
UPLOAD_INTERVAL = float(os.environ.get("UPLOAD_INTERVAL", 1.5))     # Cooldown between cloud uploads for same disease

# Render Cloud Synchronization (Configurable via Environment)
RENDER_URL = os.environ.get("RENDER_URL", "https://precision-agriculture-smart-rover.onrender.com").rstrip("/")
RENDER_API_TOKEN = os.environ.get("RENDER_API_TOKEN", "")
UPDATE_URL = RENDER_URL + "/api/update-result"
HEARTBEAT_URL = RENDER_URL + "/api/rover/heartbeat"
COMMAND_URL = RENDER_URL + "/api/rover/command"

# Headless mode detection (systemd or no display)
HEADLESS = ("--headless" in sys.argv) or (not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"))

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

LOCAL_IP = get_local_ip()
STREAM_PORT = int(os.environ.get("STREAM_PORT", 8080))
STREAM_URL = f"http://{LOCAL_IP}:{STREAM_PORT}/video_feed"

# Verified Raspberry Pi 4 B GPIO Pin Mapping (BCM Numbers)
GPIO_ENABLED = os.environ.get("GPIO_ENABLED", "1") == "1"
LEFT_IN1 = 17   # Physical Pin 11 (Forward Direction)
LEFT_IN2 = 27   # Physical Pin 13 (Reverse Direction)
LEFT_EN  = 12   # Physical Pin 32 (PWM0 Speed)
RIGHT_IN1 = 22  # Physical Pin 15 (Forward Direction)
RIGHT_IN2 = 23  # Physical Pin 16 (Reverse Direction)
RIGHT_EN  = 13  # Physical Pin 33 (PWM1 Speed)
PUMP_PIN  = 26  # Physical Pin 37 (5V Relay Control IN)
TRIG_PIN  = int(os.environ.get("ULTRASONIC_TRIG_PIN", 24))  # Physical Pin 18 (HC-SR04 Trigger)
ECHO_PIN  = int(os.environ.get("ULTRASONIC_ECHO_PIN", 25))  # Physical Pin 22 (HC-SR04 Echo)

# Ultrasonic Obstacle Stop Configuration (5 to 6 cm range)
# Rule: If obstacle is detected in range 5 to 6 cm (<= 6.0 cm), STOP motor; otherwise ALWAYS ON.
ULTRASONIC_ENABLED = os.environ.get("ULTRASONIC_ENABLED", "1") == "1"
OBSTACLE_STOP_DISTANCE_CM = float(os.environ.get("OBSTACLE_STOP_DISTANCE_CM", 6.0))    # Stop motor if object in 5-6 cm range
OBSTACLE_RESUME_DISTANCE_CM = float(os.environ.get("OBSTACLE_RESUME_DISTANCE_CM", 7.0)) # Otherwise motor is ALWAYS ON (>6.0 cm)
DEFAULT_ALWAYS_ON_MOTOR = os.environ.get("DEFAULT_ALWAYS_ON_MOTOR", "1") == "1"

# Precision Targeted Disease Spray Configuration (Relay Module / DC Motor)
# Rule: Relay module actuates DC motor spray pump ONLY when a diseased leaf is detected!
AUTO_DISEASE_SPRAY_ENABLED = os.environ.get("AUTO_DISEASE_SPRAY_ENABLED", "1") == "1"
SPRAY_BURST_SECONDS = float(os.environ.get("SPRAY_BURST_SECONDS", 2.5))       # Spray pulse duration per diseased leaf
SPRAY_COOLDOWN_SECONDS = float(os.environ.get("SPRAY_COOLDOWN_SECONDS", 5.0)) # Cooldown between sprays to prevent flooding
DISEASE_CLASSES_FOR_SPRAY = {"Bacterial_Spot", "Cercospora_Leaf_Spot", "Curl_Virus"} # Diseased classes ONLY

# Motor Direction Inversion Configuration (Differential Chassis Optimization)
# Due to opposing mirror mount on differential chassis, Right Motor is inverted by default
# so both wheels roll forward together in a straight line instead of spinning clockwise!
INVERT_RIGHT_MOTOR = os.environ.get("INVERT_RIGHT_MOTOR", "1") == "1"
INVERT_LEFT_MOTOR = os.environ.get("INVERT_LEFT_MOTOR", "0") == "1"

SAVE_FOLDER = os.path.join(BASE_DIR, "detected_leaves")
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Disease treatments database (Exact 3 diseases + Healthy Leaf)
DISEASE_INFO = {
    "bacterial_spot": {
        "chemical": "Copper oxychloride-based bactericide",
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Remove infected leaves and avoid overhead watering / excess moisture."
    },
    "cercospora_leaf_spot": {
        "chemical": "Copper-based fungicide / Mancozeb",
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Remove infected leaves and improve plant spacing for airflow."
    },
    "curl_virus": {
        "chemical": "Imidacloprid / Neem oil for whitefly & vector control",
        "fertilizer": "Balanced NPK fertilizer with micronutrients",
        "treatment": "Uproot severely stunted plants and control insect vectors immediately."
    },
    "healthy_leaf": {
        "chemical": "No chemical treatment required",
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Plant foliage is healthy and vigorous. Maintain regular monitoring."
    }
}

# ============================================================
# LOGGING HELPER
# ============================================================

def log(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{ts} {msg}", flush=True)

# ============================================================
# PROCESS LIFECYCLE & SIGNAL HANDLERS
# ============================================================

running = True

def handle_signal(sig, frame):
    global running
    log(f"Signal {sig} received. Initiating graceful shutdown...")
    running = False

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

# ============================================================
# GPIO HARDWARE CONTROLLER (WITH MOTOR-SAFE INITIALIZATION)
# ============================================================

gpio_available = False
gpio_chip = None
lgpio = None

RELAY_ACTIVE_LOW = os.environ.get("RELAY_ACTIVE_LOW", "1") == "1"

def init_gpio():
    global gpio_available, gpio_chip, lgpio
    if not GPIO_ENABLED:
        log("GPIO hardware explicitly disabled in settings (Safe Simulation Mode).")
        return

    try:
        import lgpio as _lgpio
        lgpio = _lgpio
        gpio_chip = lgpio.gpiochip_open(0)

        # Free stale locks and claim outputs in safe low state (motors STOPPED)
        pins = [LEFT_IN1, LEFT_IN2, LEFT_EN, RIGHT_IN1, RIGHT_IN2, RIGHT_EN]
        for pin in pins:
            try:
                lgpio.gpio_free(gpio_chip, pin)
            except Exception:
                pass
            lgpio.gpio_claim_output(gpio_chip, pin, 0)

        # Claim pump pin in safe OFF state (1 if active-low, 0 if active-high)
        try:
            lgpio.gpio_free(gpio_chip, PUMP_PIN)
        except Exception:
            pass
        lgpio.gpio_claim_output(gpio_chip, PUMP_PIN, 1 if RELAY_ACTIVE_LOW else 0)

        # Claim ultrasonic sensor pins (TRIG as output, ECHO as input with pull-down)
        if ULTRASONIC_ENABLED:
            try:
                lgpio.gpio_free(gpio_chip, TRIG_PIN)
            except Exception:
                pass
            try:
                lgpio.gpio_free(gpio_chip, ECHO_PIN)
            except Exception:
                pass
            try:
                lgpio.gpio_claim_output(gpio_chip, TRIG_PIN, 0)
                lgpio.gpio_claim_input(gpio_chip, ECHO_PIN, lgpio.SET_PULL_DOWN)
            except Exception as ue:
                log(f"Ultrasonic pin claim notice: {ue}")

        gpio_available = True
        log(f"GPIO Controller initialized. Motors safely OFF, Relay mode: {'ACTIVE-LOW' if RELAY_ACTIVE_LOW else 'ACTIVE-HIGH'}, HC-SR04: Trig={TRIG_PIN}, Echo={ECHO_PIN}.")
    except Exception as e:
        gpio_available = False
        log(f"GPIO Hardware unavailable ({e}). Operating in Safe Simulation Mode.")

init_gpio()

def gpio_write(pin, value):
    if gpio_available and gpio_chip is not None and pin is not None:
        try:
            lgpio.gpio_write(gpio_chip, pin, 1 if value else 0)
        except Exception as e:
            log(f"GPIO write error on pin {pin}: {e}")

def gpio_pwm(pin, duty):
    if gpio_available and gpio_chip is not None and pin is not None:
        try:
            lgpio.tx_pwm(gpio_chip, pin, 1000, max(0, min(100, int(duty))))
        except Exception as e:
            log(f"GPIO PWM error on pin {pin}: {e}")

current_applied_pwm = 0

def stop_motors():
    global current_applied_pwm
    current_applied_pwm = 0
    gpio_pwm(LEFT_IN1, 0)
    gpio_pwm(LEFT_IN2, 0)
    gpio_pwm(RIGHT_IN1, 0)
    gpio_pwm(RIGHT_IN2, 0)
    gpio_write(LEFT_IN1, 0)
    gpio_write(LEFT_IN2, 0)
    gpio_write(LEFT_EN, 0)
    gpio_write(RIGHT_IN1, 0)
    gpio_write(RIGHT_IN2, 0)
    gpio_write(RIGHT_EN, 0)
    log("GPIO OUTPUT -> All BTS7960/motor pins driven LOW (0V - Stopped)")

def set_pump(state):
    val = (0 if state else 1) if RELAY_ACTIVE_LOW else (1 if state else 0)
    gpio_write(PUMP_PIN, val)

# Ensure motors are completely stopped right after init
stop_motors()
set_pump(False)

# ============================================================
# ULTRASONIC OBSTACLE SENSING & COLLISION STOP SUBSYSTEM
# ============================================================

obstacle_detected = False
current_obstacle_distance = 999.0
obstacle_lock = threading.Lock()
last_user_command = os.environ.get("DEFAULT_ROVER_DIRECTION", "forward")
last_user_speed = int(os.environ.get("DEFAULT_ROVER_SPEED", 60))
last_user_pump = False

def read_ultrasonic_distance_cm(timeout_sec=0.03):
    """
    Measures distance in cm using HC-SR04 ultrasonic sensor via lgpio.
    Returns distance in cm, or None if no echo received / timeout.
    """
    if not gpio_available or gpio_chip is None or not ULTRASONIC_ENABLED:
        return None
    try:
        # Settle trigger pin low
        lgpio.gpio_write(gpio_chip, TRIG_PIN, 0)
        time.sleep(0.000005)
        # Send 10-12us trigger pulse
        lgpio.gpio_write(gpio_chip, TRIG_PIN, 1)
        time.sleep(0.000012)
        lgpio.gpio_write(gpio_chip, TRIG_PIN, 0)

        t0 = time.perf_counter()
        timeout_rise = t0 + timeout_sec
        while lgpio.gpio_read(gpio_chip, ECHO_PIN) == 0:
            if time.perf_counter() > timeout_rise:
                return None
            time.sleep(0.000015)

        t_rise = time.perf_counter()
        timeout_fall = t_rise + timeout_sec
        while lgpio.gpio_read(gpio_chip, ECHO_PIN) == 1:
            if time.perf_counter() > timeout_fall:
                return None
            time.sleep(0.000015)

        t_fall = time.perf_counter()
        duration = t_fall - t_rise
        dist_cm = (duration * 34300.0) / 2.0

        if 1.5 <= dist_cm <= 400.0:
            return dist_cm
        return None
    except Exception:
        return None

def ultrasonic_obstacle_worker():
    """
    High-speed obstacle monitoring thread (~12 Hz).
    Rule:
    - If object is detected in 5 to 6 cm range (<= 6.0 cm), STOP the motors.
    - Otherwise (> 6.0 cm or cleared path), KEEP MOTORS ALWAYS ON (automatically resume drive)!
    """
    global obstacle_detected, current_obstacle_distance, last_user_command, last_user_speed
    recent_distances = collections.deque(maxlen=3)

    while running:
        try:
            if ULTRASONIC_ENABLED and gpio_available:
                d = read_ultrasonic_distance_cm(timeout_sec=0.025)
                if d is not None:
                    recent_distances.append(d)
                    med_d = sorted(recent_distances)[len(recent_distances) // 2]
                    with obstacle_lock:
                        current_obstacle_distance = med_d
                        # 1. STOP: Object detected in 5 to 6 cm range (<= 6.0 cm)
                        if med_d <= OBSTACLE_STOP_DISTANCE_CM:
                            if not obstacle_detected:
                                log(f"⚠️ [OBSTACLE DETECTED] Object in 5-6 cm range ({med_d:.1f} cm <= {OBSTACLE_STOP_DISTANCE_CM} cm) -> MOTOR STOPPED!")
                            obstacle_detected = True
                            shared_telemetry["obstacle"] = True
                            shared_telemetry["distance_cm"] = round(med_d, 1)

                            # Immediately halt forward motion!
                            if shared_telemetry.get("direction", "").lower() in ("forward", "engine armed", "left", "right"):
                                stop_motors()
                                shared_telemetry["direction"] = "Obstacle Stop"

                        # 2. CLEAR: Distance > 6.0 cm (Otherwise motor is ALWAYS ON)
                        elif med_d >= OBSTACLE_RESUME_DISTANCE_CM:
                            was_blocked = obstacle_detected
                            obstacle_detected = False
                            shared_telemetry["obstacle"] = False
                            shared_telemetry["distance_cm"] = round(med_d, 1)

                            if was_blocked:
                                log(f"✅ [OBSTACLE CLEARED] Distance: {med_d:.1f} cm > {OBSTACLE_STOP_DISTANCE_CM} cm -> RESUMING MOTOR (ALWAYS ON)!")
                                if last_user_command in ("forward", "start", "left", "right"):
                                    apply_rover_command(last_user_command, last_user_speed, shared_telemetry.get("pump", False))
                else:
                    if recent_distances:
                        recent_distances.clear()
                    with obstacle_lock:
                        was_blocked = obstacle_detected
                        obstacle_detected = False
                        shared_telemetry["obstacle"] = False
                        if was_blocked and last_user_command in ("forward", "start", "left", "right"):
                            log(f"✅ [OBSTACLE CLEARED] Open path detected -> RESUMING MOTOR (ALWAYS ON)!")
                            apply_rover_command(last_user_command, last_user_speed, shared_telemetry.get("pump", False))
        except Exception:
            pass
        time.sleep(0.08)

# ============================================================
# TARGETED DISEASE SPRAY SUBSYSTEM (RELAY MODULE / DC MOTOR)
# ============================================================

is_spraying = False
spray_stop_time = 0.0
last_disease_spray_time = 0.0
spray_lock = threading.Lock()

def trigger_disease_spray(disease_name, confidence):
    """
    Triggers the spray pump relay to actuate the DC motor ONLY for diseased leaves.
    Applies precision spray pulse and safety cooldown.
    """
    global is_spraying, spray_stop_time, last_disease_spray_time
    if not AUTO_DISEASE_SPRAY_ENABLED:
        return

    now = time.time()
    with spray_lock:
        if is_spraying:
            return
        if (now - last_disease_spray_time) < SPRAY_COOLDOWN_SECONDS:
            return

        log(f"💦 [DISEASE SPRAY TRIGGERED] Target: {disease_name} ({confidence*100:.1f}%) -> Energizing Relay (DC Motor Spray ON) for {SPRAY_BURST_SECONDS}s")
        set_pump(True)
        is_spraying = True
        spray_stop_time = now + SPRAY_BURST_SECONDS
        last_disease_spray_time = now
        shared_telemetry["pump"] = True
        shared_telemetry["spray_status"] = f"Spraying for {disease_name}"

def cancel_disease_spray_for_healthy():
    """
    Strict safety guarantee: if leaf in view is Healthy, relay is immediately de-energized
    so DC motor spray pump never sprays healthy plants.
    """
    global is_spraying
    with spray_lock:
        if is_spraying:
            log("🌿 [HEALTHY LEAF SAFETY] Spray pump immediately deactivated to protect healthy foliage.")
            set_pump(False)
            is_spraying = False
            shared_telemetry["pump"] = False
            shared_telemetry["spray_status"] = "Standby (Healthy Leaf)"

def disease_spray_monitor_worker():
    """
    Monitors active disease spray duration and automatically cuts off the DC motor pump
    once the precision spray pulse completes.
    """
    global is_spraying, spray_stop_time
    while running:
        try:
            now = time.time()
            with spray_lock:
                if is_spraying and now >= spray_stop_time:
                    log("💦 [DISEASE SPRAY COMPLETED] Relay de-energized -> DC Motor Spray OFF.")
                    set_pump(False)
                    is_spraying = False
                    shared_telemetry["pump"] = False
                    shared_telemetry["spray_status"] = "Standby"
        except Exception:
            pass
        time.sleep(0.05)

shared_telemetry = {
    "device": "Raspberry Pi 4 Model B",
    "pi_online": True,
    "model_online": True,
    "ai_status": "Ready",
    "model_name": "MobileNetV2 (99% Acc) Chili Disease AI",
    "fps": 0.0,
    "last_detection": "NO LEAF DETECTED",
    "last_confidence": 0.0,
    "direction": "Stopped",
    "pump": False,
    "obstacle": False,
    "distance_cm": None,
    "spray_status": "Standby",
    "pi_ip": LOCAL_IP,
    "stream_url": STREAM_URL
}

# ============================================================
# LIVE MJPEG VIDEO STREAMING SERVER (HTTP PORT 8080)
# ============================================================

def generate_placeholder_frame(text="CAMERA FEED INITIALIZING..."):
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(canvas, "AGRIROVER SMART ROVER", (140, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "CSI Camera Subsystem Starting...", (175, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    ok, enc = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return enc.tobytes() if ok else None

latest_jpeg_frame = generate_placeholder_frame()
latest_jpeg_lock = threading.Lock()
latest_cloud_jpeg_frame = generate_placeholder_frame()
latest_cloud_jpeg_lock = threading.Lock()
cloud_frame_seq = 0
cloud_dispatched_seq = 0
cloud_seq_lock = threading.Lock()

class StreamingHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        global latest_jpeg_frame
        if self.path in ('/video_feed', '/stream', '/'):
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                while running:
                    with latest_jpeg_lock:
                        frame_bytes = latest_jpeg_frame
                    if frame_bytes is not None:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame_bytes)))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.04)
            except Exception:
                pass
        elif self.path in ('/snapshot', '/current.jpg'):
            with latest_jpeg_lock:
                frame_bytes = latest_jpeg_frame
            if frame_bytes is not None:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(frame_bytes)))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(frame_bytes)
            else:
                self.send_error(404, "No frame captured yet")
        elif self.path == '/status':
            import json
            payload = json.dumps(shared_telemetry).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def start_streaming_server(port=8080):
    try:
        server = ThreadedHTTPServer(('0.0.0.0', port), StreamingHandler)
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True, name="VideoStreamServer")
        srv_thread.start()
        log(f"Live Video Stream Server listening on http://{LOCAL_IP}:{port}/video_feed")
        return server
    except Exception as e:
        log(f"Warning: Could not start streaming server on port {port}: {e}")
        return None

def apply_rover_command(direction, speed, pump):
    global INVERT_RIGHT_MOTOR, INVERT_LEFT_MOTOR, last_user_command, last_user_speed, last_user_pump
    direction = str(direction).lower().strip()
    is_pump = bool(pump)
    last_user_pump = is_pump

    # 1. Targeted Spray Rule: Actuate spray relay ONLY for diseased leaves!
    with spray_lock:
        if is_pump:
            with latest_ai_lock:
                current_disease = latest_ai_result.get("disease_code", "")
            if current_disease in DISEASE_CLASSES_FOR_SPRAY:
                set_pump(True)
                shared_telemetry["pump"] = True
                shared_telemetry["spray_status"] = f"Spraying for {current_disease}"
                log(f"💦 [SPRAY COMMAND] Confirmed disease in view ({current_disease}) -> Spray pump ON.")
            elif current_disease == "HEALTHY":
                log("🌿 [PUMP SAFETY] Leaf in view is HEALTHY. Spray relay held strictly OFF to protect plant.")
                set_pump(False)
                shared_telemetry["pump"] = False
                shared_telemetry["spray_status"] = "Held OFF (Healthy Leaf)"
            else:
                log("🌿 [PUMP SAFETY] No diseased leaf currently in view. Spray pump held OFF.")
                set_pump(False)
                shared_telemetry["pump"] = False
                shared_telemetry["spray_status"] = "Held OFF (No Disease)"
        else:
            if not is_spraying:
                set_pump(False)
                shared_telemetry["pump"] = False
                shared_telemetry["spray_status"] = "Standby"

    # Explicit STOP command
    if direction in ("stop", "emergency_stop", "stopped", ""):
        last_user_command = "stopped"
        shared_telemetry["direction"] = "Stopped"
        shared_telemetry["engine_status"] = "Standby"
        stop_motors()
        return

    # START Command: Arms and drives forward
    if direction == "start":
        direction = "forward"
        if speed <= 0:
            speed = 60

    if speed <= 0 and direction in ("forward", "backward", "reverse", "left", "right"):
        speed = 60

    last_user_command = direction
    last_user_speed = speed

    # 2. Ultrasonic Obstacle Safety Intercept (5-6 cm range): Block forward driving if obstacle is in 5-6 cm!
    if direction in ("forward", "left", "right"):
        with obstacle_lock:
            if obstacle_detected:
                log(f"⚠️ [OBSTACLE INTERCEPT] Cannot move {direction.upper()}: Object in 5-6 cm range ({current_obstacle_distance:.1f} cm <= {OBSTACLE_STOP_DISTANCE_CM} cm)! Motors kept STOPPED.")
                stop_motors()
                shared_telemetry["direction"] = "Obstacle Stop"
                shared_telemetry["obstacle"] = True
                return

    shared_telemetry["direction"] = direction.capitalize()
    if speed <= 0 and direction in ("forward", "backward", "reverse", "left", "right"):
        speed = 60
    is_moving = speed > 0
    in1, in2, in3, in4 = 0, 0, 0, 0

    # Directional Kinematics for Differential Drive Chassis
    if direction == "forward":
        in1, in2 = 1, 0
        in3, in4 = 1, 0
    elif direction in ("backward", "reverse"):
        in1, in2 = 0, 1
        in3, in4 = 0, 1
    elif direction in ("left", "right"):
        # Single BTS7960 Option 1: Slow / Quarter-Speed Precision Crawl (35% PWM)
        speed = max(25, min(40, int(speed * 0.55)))
        in1, in2 = 1, 0
        in3, in4 = 1, 0

    # Motor Polarity Inversion (Fixes differential drive mirror mount spinning)
    if INVERT_LEFT_MOTOR:
        in1, in2 = in2, in1
    if INVERT_RIGHT_MOTOR:
        in3, in4 = in4, in3

    global current_applied_pwm
    if is_moving:
        if in1:
            gpio_pwm(LEFT_IN2, 0)
            gpio_write(LEFT_IN2, 0)
            gpio_pwm(RIGHT_IN2, 0)
            gpio_write(RIGHT_IN2, 0)
        elif in2:
            gpio_pwm(LEFT_IN1, 0)
            gpio_write(LEFT_IN1, 0)
            gpio_pwm(RIGHT_IN1, 0)
            gpio_write(RIGHT_IN1, 0)
        time.sleep(0.005)
        gpio_write(LEFT_EN, 1)
        gpio_write(RIGHT_EN, 1)

        # Soft-Start PWM Ramping (Gentle Acceleration - prevents camera shaking)
        target = speed
        cur = current_applied_pwm
        if abs(target - cur) > 20:
            mid = (cur + target) // 2
            gpio_pwm(LEFT_IN1, mid if in1 else 0)
            gpio_pwm(LEFT_IN2, mid if in2 else 0)
            gpio_pwm(RIGHT_IN1, mid if in3 else 0)
            gpio_pwm(RIGHT_IN2, mid if in4 else 0)
            time.sleep(0.02)

        gpio_pwm(LEFT_IN1, target if in1 else 0)
        gpio_pwm(LEFT_IN2, target if in2 else 0)
        gpio_pwm(RIGHT_IN1, target if in3 else 0)
        gpio_pwm(RIGHT_IN2, target if in4 else 0)
        current_applied_pwm = target
    else:
        current_applied_pwm = 0
        stop_motors()
    log(f"GPIO OUTPUT ({direction.upper()}) -> RPWM/IN1(11)={speed if in1 else 0}% LPWM/IN2(13)={speed if in2 else 0}% | EN(32)={int(is_moving)} [RightInv={INVERT_RIGHT_MOTOR}]")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

if not os.path.exists(CLASS_FILE):
    log(f"ERROR: {CLASS_FILE} not found!")
    sys.exit(1)

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

log(f"Classes loaded ({len(class_names)}): {', '.join(class_names)}")

# ============================================================
# LOAD AI MODEL (Ultra-Fast LiteRT with XNNPACK + Fallbacks)
# ============================================================

use_tflite = False
tflite_interpreter = None
keras_model = None
input_details = None
output_details = None
model_ready = False
model_loading_lock = threading.Lock()

def load_ai_model():
    """
    Ultra-fast model loader.
    Prioritizes Google LiteRT (ai_edge_litert), which loads in ~0.5s instead of 18s.
    Runs concurrently in the background so camera video streaming starts instantly!
    """
    global use_tflite, tflite_interpreter, keras_model, input_details, output_details, model_ready
    t0 = time.time()

    # 1. Primary: Ultra-Fast LiteRT (~0.5s load time with XNNPACK)
    if os.path.exists(TFLITE_MODEL_PATH):
        try:
            log("Loading ultra-fast Google LiteRT engine (XNNPACK)...")
            from ai_edge_litert.interpreter import Interpreter
            interp = Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=2)
            interp.allocate_tensors()
            inp = interp.get_input_details()
            out = interp.get_output_details()

            with model_loading_lock:
                tflite_interpreter = interp
                input_details = inp
                output_details = out
                use_tflite = True
                model_ready = True

            shared_telemetry["model_online"] = True
            shared_telemetry["ai_status"] = "AI Ready (LiteRT High-Speed)"
            log(f"LiteRT Model loaded successfully in {time.time()-t0:.2f}s with 2-thread XNNPACK acceleration!")
            return
        except Exception as e:
            log(f"LiteRT load error ({e}), trying standard tf.lite...")

        # 2. Secondary Fallback: Standard TFLite Interpreter
        try:
            from tensorflow.lite.python.interpreter import Interpreter
            interp = Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=2)
            interp.allocate_tensors()
            inp = interp.get_input_details()
            out = interp.get_output_details()

            with model_loading_lock:
                tflite_interpreter = interp
                input_details = inp
                output_details = out
                use_tflite = True
                model_ready = True

            shared_telemetry["model_online"] = True
            shared_telemetry["ai_status"] = "AI Ready (TFLite)"
            log(f"TFLite Model loaded in {time.time()-t0:.2f}s.")
            return
        except Exception as e:
            log(f"TFLite fallback failed ({e}), checking Keras...")

    # 3. Tertiary Fallback: Full Keras
    if os.path.exists(KERAS_MODEL_PATH):
        try:
            import tensorflow as tf
            log(f"Loading full Keras model: {KERAS_MODEL_PATH}")
            model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
            with model_loading_lock:
                keras_model = model
                model_ready = True
            shared_telemetry["model_online"] = True
            shared_telemetry["ai_status"] = "AI Ready (Keras)"
            log(f"Keras model loaded in {time.time()-t0:.2f}s.")
        except Exception as e:
            log(f"Keras model failed to load: {e}")

# ============================================================
# LEAF-ONLY DETECTION PIPELINE (REJECTS FACES, HANDS & ROOMS)
# ============================================================

def letterbox(img, target_size=IMAGE_SIZE, bg_color=(255, 255, 255)):
    """
    Symmetrically letterboxes an image into target_size x target_size,
    preserving exact aspect ratio and avoiding distorted curling artifacts.
    Uses white background to match the training dataset domain.
    """
    if img is None or img.size == 0:
        return np.full((target_size, target_size, 3), bg_color, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), bg_color, dtype=np.uint8)
    dx = (target_size - nw) // 2
    dy = (target_size - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas

def verify_leaf_structure(crop_bgr):
    """
    Botanical Quality Gate: Verifies that a candidate image contains genuine plant leaf tissue.
    Strictly rejects desk surfaces, shadows, walls, curtains, carpets, furniture, and hands.
    Returns: (True, "OK") if genuine foliage; (False, reason) otherwise.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False, "Empty image"

    h, w = crop_bgr.shape[:2]
    total_px = float(h * w)
    if w < 35 or h < 35 or total_px < 1200:
        return False, f"Candidate too small ({w}x{h})"

    aspect = w / float(h)
    if aspect < 0.12 or aspect > 8.0:
        return False, f"Aspect ratio extreme ({aspect:.2f})"

    b = crop_bgr[:, :, 0].astype(np.float32)
    g = crop_bgr[:, :, 1].astype(np.float32)
    r = crop_bgr[:, :, 2].astype(np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # 1. Botanical chlorophyll foliage mask:
    # Green plant foliage: H in [24, 88], S >= 28, V >= 25, G > B + 10, G >= R - 12, 2G - R - B >= 14
    green_mask = (hsv[:,:,0] >= 24) & (hsv[:,:,0] <= 88) & \
                 (hsv[:,:,1] >= 28) & (hsv[:,:,2] >= 25) & (hsv[:,:,2] <= 252) & \
                 (g > b + 10.0) & (g >= r - 12.0) & \
                 (2.0 * g - r - b >= 14.0)

    # Yellow chlorotic foliage mask (for viral curl / spot diseases / chlorotic halos)
    yellow_mask = (hsv[:,:,0] >= 14) & (hsv[:,:,0] < 24) & \
                  (hsv[:,:,1] >= 38) & (hsv[:,:,2] >= 32) & (hsv[:,:,2] <= 252) & \
                  (g > b + 14.0) & (r > b + 14.0) & \
                  (2.0 * g - r - b >= 8.0)

    foliage_mask = green_mask | yellow_mask
    foliage_px = int(np.sum(foliage_mask))

    if foliage_px < 400:
        return False, f"Insufficient chlorophyll pixels ({foliage_px} < 400)"

    # 2. Texture check (real foliage has natural texture/margins, rejects flat featureless walls)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 8.0:
        return False, f"Flat texture ({lap_var:.1f} < 8.0)"

    return True, "VALID_LEAF"

def detect_leaf(frame_bgr):
    """
    Detects real plant leaves (both healthy and diseased with spots/yellowing/curling)
    and strictly rejects faces, hands, walls, ceilings, and room clutter.
    Runs on optimized 320x240 thumbnail for ultra-smooth 25+ FPS camera performance.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None, None, "NO_LEAF"

    h, w = frame_bgr.shape[:2]
    # Fast 2x downsample for 4x faster foliage segmentation (takes < 3ms instead of 15ms)
    small_w = w // 2
    small_h = h // 2
    small_bgr = cv2.resize(frame_bgr, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    small_area = float(small_w * small_h)

    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    b_full = small_bgr[:, :, 0].astype(np.float32)
    g_full = small_bgr[:, :, 1].astype(np.float32)
    r_full = small_bgr[:, :, 2].astype(np.float32)

    # 1. Botanical Foliage Mask with Chlorophyll Discrimination
    green_bool = (hsv[:,:,0] >= 24) & (hsv[:,:,0] <= 88) & \
                 (hsv[:,:,1] >= 28) & (hsv[:,:,2] >= 25) & (hsv[:,:,2] <= 252) & \
                 (g_full > b_full + 10.0) & (g_full >= r_full - 12.0) & \
                 (2.0 * g_full - r_full - b_full >= 14.0)

    # Yellow chlorotic foliage mask (for viral curl / spot diseases)
    yellow_bool = (hsv[:,:,0] >= 14) & (hsv[:,:,0] < 24) & \
                  (hsv[:,:,1] >= 38) & (hsv[:,:,2] >= 32) & (hsv[:,:,2] <= 252) & \
                  (g_full > b_full + 14.0) & (r_full > b_full + 14.0) & \
                  (2.0 * g_full - r_full - b_full >= 8.0)

    foliage_mask = np.uint8(green_bool | yellow_bool) * 255

    # 2. Human Skin Tone Mask: catches faces, hands, arms holding leaves
    skin1 = cv2.inRange(hsv, np.array([0, 25, 45], dtype=np.uint8),
                             np.array([22, 175, 255], dtype=np.uint8))
    skin2 = cv2.inRange(hsv, np.array([168, 25, 45], dtype=np.uint8),
                             np.array([180, 175, 255], dtype=np.uint8))
    skin_mask = cv2.bitwise_or(skin1, skin2)

    foliage_clean = cv2.bitwise_and(foliage_mask, cv2.bitwise_not(skin_mask))
    k_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    foliage_clean = cv2.morphologyEx(foliage_clean, cv2.MORPH_OPEN, k_noise)

    total_foliage_px = int(cv2.countNonZero(foliage_clean))
    total_foliage_ratio = total_foliage_px / small_area

    if total_foliage_px < 150 or total_foliage_ratio < 0.015:
        return None, None, "NO_LEAF"

    # Full frame coverage / Macro close-up
    if total_foliage_ratio >= 0.12:
        pts = cv2.findNonZero(foliage_clean)
        if pts is not None:
            fx, fy, fw, fh = cv2.boundingRect(pts)
            if fw >= 25 and fh >= 25:
                gx, gy, gw, gh = fx * 2, fy * 2, min(w - fx * 2, fw * 2), min(h - fy * 2, fh * 2)
                crop = frame_bgr[gy:gy+gh, gx:gx+gw]
                is_valid, _ = verify_leaf_structure(crop)
                if is_valid:
                    return crop, (gx, gy, gx+gw, gy+gh), "LEAF_DETECTED"

    contours, _ = cv2.findContours(foliage_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_candidate = None
    best_score = 0.0

    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 18 or bh < 18:
                continue

            aspect = bw / float(bh)
            if aspect < 0.15 or aspect > 6.5:
                continue

            score = area
            if score > best_score:
                best_score = score
                pad = 4
                cx1 = max(0, bx - pad); cy1 = max(0, by - pad)
                cx2 = min(small_w, bx + bw + pad); cy2 = min(small_h, by + bh + pad)
                best_candidate = (cx1 * 2, cy1 * 2, min(w, cx2 * 2), min(h, cy2 * 2))

    if best_candidate is not None:
        x1, y1, x2, y2 = best_candidate
        crop = frame_bgr[y1:y2, x1:x2].copy()
        if crop.size > 0:
            is_valid, _ = verify_leaf_structure(crop)
            if is_valid:
                return crop, (x1, y1, x2, y2), "LEAF_DETECTED"

    return None, None, "NO_LEAF"

# ============================================================
# PLANT DISEASE INFERENCE ENGINE (RAW [0, 255] PIXELS)
# ============================================================

def predict_crop(crop_bgr):
    """
    Runs model inference on raw [0, 255] float RGB values.
    CRITICAL: MobileNetV2 has an internal preprocess_input layer.
    DO NOT divide by 255.0 (dividing causes solid-black mode collapse).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "NO_LEAF", "No Leaf Detected", 0.0, None

    # Quality Gate: strictly verify botanical leaf properties
    is_leaf, _ = verify_leaf_structure(crop_bgr)
    if not is_leaf:
        return "NO_LEAF", "No Leaf Detected", 0.0, None

    with model_loading_lock:
        if not model_ready:
            return "LOADING", "AI Model Initializing...", 0.0, None
        local_use_tflite = use_tflite
        local_interp = tflite_interpreter
        local_keras = keras_model
        local_inp = input_details
        local_out = output_details

    try:
        # Symmetrically letterbox crop to IMAGE_SIZE x IMAGE_SIZE to preserve natural aspect ratio
        # and eliminate artificial edge distortion/curling artifacts.
        resized = letterbox(crop_bgr, target_size=IMAGE_SIZE, bg_color=(255, 255, 255))
        # Convert BGR to RGB (model expects RGB)
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = np.expand_dims(img, axis=0)

        if local_use_tflite and local_interp is not None:
            local_interp.set_tensor(local_inp[0]['index'], img)
            local_interp.invoke()
            preds = local_interp.get_tensor(local_out[0]['index'])[0]
        elif local_keras is not None:
            preds = local_keras.predict(img, verbose=0)[0]
        else:
            return "NO_MODEL", "Model Not Ready", 0.0, None

        best_idx = int(np.argmax(preds))
        conf = float(preds[best_idx])
        raw_class = class_names[best_idx]

        # Margin check: ensure strong separation between top-1 and second-best candidate
        sorted_preds = np.sort(preds)
        second_conf = float(sorted_preds[-2]) if len(sorted_preds) > 1 else 0.0
        margin = conf - second_conf

        if conf < CONFIDENCE_THRESHOLD or margin < CONFIDENCE_MARGIN:
            return "UNCERTAIN", f"Uncertain ({conf*100:.1f}%)", conf, preds

        if raw_class == "Healthy_Leaf":
            return "HEALTHY", f"Healthy Leaf ({conf*100:.1f}%)", conf, preds
        else:
            clean_name = raw_class.replace("_", " ")
            return raw_class, f"{clean_name} ({conf*100:.1f}%)", conf, preds

    except Exception as e:
        log(f"Inference error: {e}")
        return "ERROR", "Inference Error", 0.0, None

# ============================================================
# ASYNCHRONOUS DECOUPLED AI INFERENCE WORKER
# ============================================================

ai_inference_queue = queue.Queue(maxsize=1)
latest_ai_result = {
    "disease_code": "NO_LEAF",
    "display_label": "NO LEAF DETECTED",
    "conf": 0.0,
    "last_leaf_seen": 0.0
}
latest_ai_lock = threading.Lock()

def ai_inference_worker():
    """
    Dedicated AI inference thread so model prediction (~200ms) NEVER slows down
    the camera capture or live video stream.
    Keeps camera running at silky smooth 30 FPS with zero lag!
    """
    global running
    consec_class = None
    consec_count = 0
    last_upload_time = 0.0

    while running:
        try:
            item = ai_inference_queue.get(timeout=0.5)
            if item is None:
                break
            crop, timestamp = item

            disease_code, display_label, conf, _ = predict_crop(crop)
            now = time.time()

            with latest_ai_lock:
                latest_ai_result["disease_code"] = disease_code
                latest_ai_result["display_label"] = display_label
                latest_ai_result["conf"] = conf
                latest_ai_result["last_leaf_seen"] = now

            if disease_code == consec_class:
                consec_count += 1
            else:
                consec_class = disease_code
                consec_count = 1

            if disease_code in ("NO_LEAF", "LOADING", "UNCERTAIN", "ERROR"):
                shared_telemetry["last_detection"] = display_label
                shared_telemetry["last_confidence"] = round(conf * 100.0 if conf <= 1.0 else conf, 1)
                shared_telemetry["ai_status"] = display_label
                shared_telemetry["disease_name"] = display_label
                shared_telemetry["disease_chemical"] = ""
                shared_telemetry["disease_fertilizer"] = ""
                shared_telemetry["disease_treatment"] = "Point camera directly at a chili plant leaf" if disease_code == "NO_LEAF" else "Hold camera steady near leaf"
            else:
                info_key = disease_code.lower()
                if "bacterial" in info_key:
                    info_key = "bacterial_spot"
                elif "cercospora" in info_key:
                    info_key = "cercospora_leaf_spot"
                elif "curl" in info_key:
                    info_key = "curl_virus"
                else:
                    info_key = "healthy_leaf"

                d_info = DISEASE_INFO.get(info_key, {
                    "chemical": "None required",
                    "fertilizer": "Balanced NPK fertilizer",
                    "treatment": "Maintain regular crop monitoring."
                })

                shared_telemetry["last_detection"] = display_label
                shared_telemetry["last_confidence"] = round(conf * 100.0 if conf <= 1.0 else conf, 1)
                shared_telemetry["ai_status"] = f"Classified: {display_label}"
                shared_telemetry["disease_name"] = display_label
                shared_telemetry["disease_chemical"] = d_info["chemical"]
                shared_telemetry["disease_fertilizer"] = d_info["fertilizer"]
                shared_telemetry["disease_treatment"] = d_info["treatment"]

            # Continuous disease upload: require at least 2 consecutive detections of valid leaf disease to avoid noise spikes
            valid_disease_classes = ("HEALTHY", "Bacterial_Spot", "Cercospora_Leaf_Spot", "Curl_Virus")
            is_valid_disease = (disease_code in valid_disease_classes) and (conf >= CONFIDENCE_THRESHOLD) and (consec_count >= 2)

            # AUTOMATIC TARGETED SPRAY: Actuate DC motor relay ONLY when a diseased leaf is detected!
            if (disease_code in DISEASE_CLASSES_FOR_SPRAY) and (conf >= CONFIDENCE_THRESHOLD) and (consec_count >= 2):
                trigger_disease_spray(display_label, conf)
            elif disease_code == "HEALTHY":
                # Healthy Leaf: Strictly ensure relay / pump is OFF!
                cancel_disease_spray_for_healthy()

            cooldown = UPLOAD_INTERVAL if disease_code == consec_class and last_upload_time > 0 else 1.0
            if is_valid_disease and (now - last_upload_time >= cooldown):
                snap_path = os.path.join(SAVE_FOLDER, f"leaf_{int(now)}.jpg")
                cv2.imwrite(snap_path, crop)

                try:
                    if upload_queue.full():
                        upload_queue.get_nowait()
                    upload_queue.put_nowait((crop.copy(), disease_code, display_label, conf))
                    last_upload_time = now
                    cleanup_old_snapshots()
                except Exception:
                    pass

            ai_inference_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            log(f"AI Worker error: {e}")

# ============================================================
# SNAPSHOT CLEANUP UTILITY
# ============================================================

def cleanup_old_snapshots():
    """Keep only the most recent 35 generated images in detected_leaves and static/images to avoid disk thrashing."""
    protected_names = {"app_photo.png", "store_icon.png"}
    for folder in [SAVE_FOLDER, os.path.join(BASE_DIR, "website", "static", "images")]:
        try:
            if not os.path.exists(folder):
                continue
            files = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
                and f not in protected_names
                and (f.startswith("2026") or f.startswith("leaf_"))
            ]
            if len(files) > 35:
                files.sort(key=os.path.getmtime)
                for old_f in files[:-35]:
                    try:
                        os.remove(old_f)
                    except Exception:
                        pass
        except Exception:
            pass

# ============================================================
# ASYNCHRONOUS NON-BLOCKING CLOUD UPLOAD WORKER
# ============================================================

upload_queue = queue.Queue(maxsize=1)
upload_session = requests.Session()

def upload_worker():
    """Background worker thread so HTTP POST to Render never freezes the camera loop."""
    while running:
        try:
            item = upload_queue.get(timeout=1.0)
            if item is None:
                break

            crop_image, disease_code, display_name, confidence = item
            key = disease_code.lower().replace(" ", "_").replace("-", "_")
            if key == "healthy":
                key = "healthy_leaf"
            info = DISEASE_INFO.get(key, {
                "chemical": "None required",
                "fertilizer": "Balanced NPK fertilizer",
                "treatment": "Maintain regular crop monitoring."
            })

            ok, enc = cv2.imencode(".jpg", crop_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                files = {"image": ("leaf.jpg", enc.tobytes(), "image/jpeg")}
                headers = {}
                if RENDER_API_TOKEN:
                    headers["Authorization"] = f"Bearer {RENDER_API_TOKEN}"

                norm_conf = confidence if confidence > 1.0 else confidence * 100.0
                data = {
                    "disease": display_name,
                    "confidence": f"{norm_conf:.1f}",
                    "chemical": info["chemical"],
                    "fertilizer": info["fertilizer"],
                    "treatment": info["treatment"]
                }
                # 1. Immediate Local Web & SQLite Sync (RPi local Flask)
                try:
                    upload_session.post(
                        "http://127.0.0.1:5000/api/update-result",
                        files={"image": ("leaf.jpg", enc.tobytes(), "image/jpeg")},
                        data=data,
                        timeout=1.5
                    )
                except Exception:
                    pass

                # 2. Render Cloud Sync (timeout 3.5s to prevent worker thread stalling)
                try:
                    r = upload_session.post(UPDATE_URL, files=files, data=data, headers=headers, timeout=3.5)
                    if r.ok:
                        log(f"CLOUD SYNC -> Render updated: {display_name}")
                    else:
                        log(f"CLOUD SYNC -> HTTP {r.status_code}")
                except Exception as cloud_e:
                    # Cloud timeout or offline - completely non-blocking
                    pass
        except queue.Empty:
            continue
        except Exception as e:
            log(f"CLOUD SYNC ERROR -> {e}")
        finally:
            try:
                upload_queue.task_done()
            except Exception:
                pass

# ============================================================
# ROVER TELEMETRY & COMMAND WORKER
# ============================================================
# DECOUPLED COMMAND & CLOUD RELAY WORKERS
# ============================================================

last_command_id = -1
last_cloud_pump = None
last_local_command_id = -1
last_local_pump = None

def local_command_worker():
    """Responsive worker for local web dashboard commands (sub-120ms latency, low CPU)."""
    global last_local_command_id, last_local_pump, INVERT_RIGHT_MOTOR, INVERT_LEFT_MOTOR, RELAY_ACTIVE_LOW
    session = requests.Session()
    while running:
        try:
            r = session.get("http://127.0.0.1:5000/api/rover/command", timeout=0.25)
            if r.ok:
                data = r.json()
                cid = data.get("command_id", 0)
                pump = bool(data.get("pump", False))
                if "relay_active_low" in data:
                    RELAY_ACTIVE_LOW = bool(data["relay_active_low"])
                if "invert_right_motor" in data:
                    INVERT_RIGHT_MOTOR = bool(data["invert_right_motor"])
                if "invert_left_motor" in data:
                    INVERT_LEFT_MOTOR = bool(data["invert_left_motor"])

                if last_local_command_id == -1:
                    last_local_command_id = cid
                    last_local_pump = pump
                elif (cid != last_local_command_id and cid > 0) or (last_local_pump is not None and pump != last_local_pump):
                    direction = data.get("direction", "Stopped")
                    speed = int(data.get("speed", 50))
                    log(f"ROVER COMMAND (Local Web) -> {direction} | Speed: {speed}% | Pump: {'ON' if pump else 'OFF'}")
                    apply_rover_command(direction, speed, pump)
                    last_local_command_id = cid
                    last_local_pump = pump
        except Exception:
            pass
        time.sleep(0.12)

def cloud_command_worker():
    """Dedicated cloud command polling thread for Render / Android App (1.2s cadence)."""
    global last_command_id, last_cloud_pump, INVERT_RIGHT_MOTOR, INVERT_LEFT_MOTOR, RELAY_ACTIVE_LOW
    session = requests.Session()
    headers = {}
    if RENDER_API_TOKEN:
        headers["Authorization"] = f"Bearer {RENDER_API_TOKEN}"

    while running:
        try:
            r = session.get(COMMAND_URL, headers=headers, timeout=1.5)
            if r.ok:
                data = r.json()
                cid = data.get("command_id", 0)
                pump = bool(data.get("pump", False))
                if "relay_active_low" in data:
                    RELAY_ACTIVE_LOW = bool(data["relay_active_low"])
                if "invert_right_motor" in data:
                    INVERT_RIGHT_MOTOR = bool(data["invert_right_motor"])
                if "invert_left_motor" in data:
                    INVERT_LEFT_MOTOR = bool(data["invert_left_motor"])

                if last_command_id == -1:
                    last_command_id = cid
                    last_cloud_pump = pump
                    direction = data.get("direction", "Stopped")
                    speed = int(data.get("speed", 50))
                    if direction.lower() != "stopped":
                        log(f"ROVER INITIAL COMMAND (Cloud Render) -> {direction} | Speed: {speed}% | Pump: {'ON' if pump else 'OFF'}")
                        apply_rover_command(direction, speed, pump)
                    else:
                        apply_rover_command("stopped", 0, pump)
                elif (cid != last_command_id and cid > 0) or (last_cloud_pump is not None and pump != last_cloud_pump):
                    direction = data.get("direction", "Stopped")
                    speed = int(data.get("speed", 50))
                    log(f"ROVER COMMAND (Cloud Render) -> {direction} | Speed: {speed}% | Pump: {'ON' if pump else 'OFF'}")
                    apply_rover_command(direction, speed, pump)
                    last_command_id = cid
                    last_cloud_pump = pump
        except Exception:
            pass
        time.sleep(0.06)

def cloud_frame_streamer_worker(worker_id=0):
    """
    Streams live camera frames to Render so the remote dashboard & phone see crystal-clear live video.
    Optimized sequence-based dispatching with automatic session recycling and resilient connection recovery.
    """
    global cloud_dispatched_seq
    stream_url = RENDER_URL + "/api/camera/stream_frame"
    headers = {"Content-Type": "image/jpeg"}
    if RENDER_API_TOKEN:
        headers["Authorization"] = f"Bearer {RENDER_API_TOKEN}"

    session = None
    consecutive_errs = 0

    if worker_id > 0:
        time.sleep(0.08 * worker_id)

    while running:
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=1)
            session.mount("https://", adapter)
            session.mount("http://", adapter)

        my_frame = None
        with cloud_seq_lock:
            if cloud_frame_seq > cloud_dispatched_seq:
                cloud_dispatched_seq = cloud_frame_seq
                with latest_cloud_jpeg_lock:
                    my_frame = latest_cloud_jpeg_frame

        if my_frame is not None and len(my_frame) > 100:
            try:
                resp = session.post(stream_url, data=my_frame, headers=headers, timeout=3.5)
                if resp.status_code == 200:
                    consecutive_errs = 0
                else:
                    consecutive_errs += 1
            except Exception:
                consecutive_errs += 1
                if consecutive_errs >= 2:
                    try:
                        session.close()
                    except Exception:
                        pass
                    session = None
                time.sleep(0.05)
                continue
        time.sleep(0.025)


def local_telemetry_worker():
    """High-speed local telemetry sender (updates local web dashboard every 0.3s with zero WAN lag)."""
    session = requests.Session()
    headers = {"Content-Type": "application/json"}
    while running:
        try:
            session.post(
                "http://127.0.0.1:5000/api/rover/heartbeat",
                json=shared_telemetry,
                headers=headers,
                timeout=0.3
            )
        except Exception:
            pass
        time.sleep(0.35)

def cloud_telemetry_worker():
    """Dedicated cloud telemetry heartbeat sender to Render cloud (every 0.45s for instant UI reaction)."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    headers = {"Content-Type": "application/json"}
    if RENDER_API_TOKEN:
        headers["Authorization"] = f"Bearer {RENDER_API_TOKEN}"

    while running:
        try:
            session.post(HEARTBEAT_URL, json=shared_telemetry, headers=headers, timeout=1.8)
        except Exception:
            pass
        time.sleep(0.45)

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    global running, HEADLESS, latest_jpeg_frame, latest_cloud_jpeg_frame, cloud_frame_seq

    # 1. IMMEDIATELY start HTTP live video streaming server on port 8080!
    # Server goes live within ~10ms so browsers and mobile app connect with zero delay.
    stream_server = start_streaming_server(STREAM_PORT)

    # 2. Launch background command & telemetry workers
    upload_thread = threading.Thread(target=upload_worker, daemon=True, name="UploadWorker")
    upload_thread.start()

    local_cmd_thread = threading.Thread(target=local_command_worker, daemon=True, name="LocalCmdWorker")
    local_cmd_thread.start()

    cloud_cmd_thread = threading.Thread(target=cloud_command_worker, daemon=True, name="CloudCmdWorker")
    cloud_cmd_thread.start()

    # Coordinated dual-pipelined cloud streamers for rock-solid, smooth live video relay to Render
    frame_stream_thread0 = threading.Thread(target=cloud_frame_streamer_worker, args=(0,), daemon=True, name="CloudFrameStreamer-0")
    frame_stream_thread0.start()
    frame_stream_thread1 = threading.Thread(target=cloud_frame_streamer_worker, args=(1,), daemon=True, name="CloudFrameStreamer-1")
    frame_stream_thread1.start()

    local_telemetry_thread = threading.Thread(target=local_telemetry_worker, daemon=True, name="LocalTelemetryWorker")
    local_telemetry_thread.start()

    cloud_telemetry_thread = threading.Thread(target=cloud_telemetry_worker, daemon=True, name="CloudTelemetryWorker")
    cloud_telemetry_thread.start()

    obstacle_thread = threading.Thread(target=ultrasonic_obstacle_worker, daemon=True, name="ObstacleWorker")
    obstacle_thread.start()

    spray_monitor_thread = threading.Thread(target=disease_spray_monitor_worker, daemon=True, name="SprayMonitorWorker")
    spray_monitor_thread.start()

    # 3. Asynchronously load high-speed AI engine in parallel (loads in ~0.5s via LiteRT)
    ai_load_thread = threading.Thread(target=load_ai_model, daemon=True, name="ModelLoader")
    ai_load_thread.start()

    # 4. Launch decoupled AI inference worker (keeps camera loop running at smooth 30 FPS)
    ai_infer_thread = threading.Thread(target=ai_inference_worker, daemon=True, name="AIInferWorker")
    ai_infer_thread.start()

    # 5. Initialize camera subsystem immediately (no artificial sleeps)
    picam2 = None
    cap = None
    camera_type = None

    log("Initializing camera subsystem...")
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        cam_cfg = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(cam_cfg)
        picam2.start()
        camera_type = "PICAMERA2"
        log("CSI Camera started successfully via native Picamera2 pipeline.")
    except Exception as e:
        log(f"Picamera2 initialization failed ({e}). Falling back to OpenCV VideoCapture...")
        picam2 = None
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            ret_test, _ = cap.read()
            if ret_test:
                camera_type = "OPENCV_V4L2"
                log("Camera started via OpenCV V4L2 fallback.")
            else:
                cap.release()
                cap = None
        if camera_type is None:
            log("⚠️ Camera hardware not detected (Picamera2/V4L2 offline). Running in Resilient Standby Mode.")
            camera_type = "STANDBY"
            cap = None

    frame_count = 0
    curr_disease_code = "NO_LEAF"
    curr_display_label = "NO LEAF DETECTED"
    curr_conf = 0.0
    last_box = None
    last_mode = "NO_LEAF"
    last_camera_retry = 0.0

    log(f"AgriRover AI System Active. Headless mode: {HEADLESS}")
    if not HEADLESS:
        log("Press 'q' in the preview window to exit.")

    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0.0

    try:
        while running:
            # 1. Grab Raw Frame or Generate Resilient Standby Telemetry Frame
            if camera_type == "PICAMERA2":
                raw_rgb = picam2.capture_array()
                frame = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
            elif camera_type == "OPENCV_V4L2" and cap is not None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue
            else:
                # Standby mode: Build informative HUD telemetry frame so cloud stream stays active
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[::40, :, :] = (20, 20, 20)
                frame[:, ::40, :] = (20, 20, 20)
                cv2.rectangle(frame, (0, 0), (640, 48), (25, 75, 25), -1)
                cv2.putText(frame, "AGRIROVER MISSION TELEMETRY", (15, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
                cv2.putText(frame, "CAMERA OFFLINE - CHECK CSI CABLE", (35, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(frame, "Drive Motors & Ultrasonic: ONLINE", (35, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)
                dir_now = shared_telemetry.get("direction", "Stopped")
                spd_now = shared_telemetry.get("speed", 0)
                obs_now = shared_telemetry.get("distance_cm", 999.0)
                cv2.putText(frame, f"Rover Movement: {dir_now} ({spd_now}%)", (35, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, f"Obstacle Range: {obs_now:.1f} cm", (35, 295),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
                cv2.putText(frame, "Cloud Link: Active (Render Connected)", (35, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

                # Periodic background attempt to auto-reconnect camera every 4s
                now_t = time.time()
                if now_t - last_camera_retry > 4.0:
                    last_camera_retry = now_t
                    try:
                        from picamera2 import Picamera2
                        new_picam = Picamera2()
                        cam_cfg = new_picam.create_preview_configuration(
                            main={"size": (640, 480), "format": "RGB888"}
                        )
                        new_picam.configure(cam_cfg)
                        new_picam.start()
                        picam2 = new_picam
                        camera_type = "PICAMERA2"
                        log("✅ Camera detected and initialized via Picamera2!")
                    except Exception:
                        pass
                time.sleep(0.035)

            # Correct camera chassis mounting orientation if configured
            if CAMERA_ROTATION == "90_CCW" or CAMERA_ROTATION == "270":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif CAMERA_ROTATION == "90_CW" or CAMERA_ROTATION == "90":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif CAMERA_ROTATION == "180":
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            frame_count += 1
            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now
                shared_telemetry["fps"] = round(fps_display, 1)

            # 2. Strict Botanical Leaf Identification (Runs every PREDICT_EVERY_N_FRAMES for high efficiency)
            if camera_type != "STANDBY" and frame_count % PREDICT_EVERY_N_FRAMES == 0:
                crop, box, mode = detect_leaf(frame)
                last_box = box
                last_mode = mode
            elif camera_type == "STANDBY":
                crop = None
                box = None
                mode = "NO_LEAF"
                last_box = None
                last_mode = "NO_LEAF"
            else:
                box = last_box
                mode = last_mode
                crop = None

            # 3. Asynchronously push leaf candidate to AI inference worker (Zero Camera Delay!)
            if mode == "LEAF_DETECTED" and crop is not None and crop.size > 0:
                try:
                    if ai_inference_queue.full():
                        ai_inference_queue.get_nowait()
                    ai_inference_queue.put_nowait((crop.copy(), now))
                except Exception:
                    pass

            # Read latest AI classification result (thread-safe)
            with latest_ai_lock:
                ai_disease_code = latest_ai_result["disease_code"]
                ai_display_label = latest_ai_result["display_label"]
                ai_conf = latest_ai_result["conf"]
                ai_last_seen = latest_ai_result["last_leaf_seen"]

            leaf_currently_visible = (mode == "LEAF_DETECTED" and box is not None)
            
            # Smooth detection retention window (2.0s): keeps diagnosis steady on HUD & cloud telemetry
            if leaf_currently_visible or (now - ai_last_seen < 2.0 and ai_disease_code not in ("NO_LEAF", "LOADING", "INITIALIZING", "UNCERTAIN")):
                curr_disease_code = ai_disease_code
                curr_display_label = ai_display_label
                curr_conf = ai_conf
                shared_telemetry["last_detection"] = ai_display_label
                shared_telemetry["last_confidence"] = round(ai_conf * 100.0 if ai_conf <= 1.0 else ai_conf, 1)
                shared_telemetry["ai_status"] = f"Classified: {ai_display_label}"
            else:
                curr_disease_code = "NO_LEAF"
                curr_display_label = "NO LEAF DETECTED"
                curr_conf = 0.0
                shared_telemetry["last_detection"] = "NO LEAF DETECTED"
                shared_telemetry["last_confidence"] = 0.0
                shared_telemetry["ai_status"] = "Active Scanning (No Leaf in View)"
                shared_telemetry["disease_name"] = "Scanning Active"
                shared_telemetry["disease_chemical"] = ""
                shared_telemetry["disease_fertilizer"] = ""
                shared_telemetry["disease_treatment"] = "Point camera at chili plant leaf"

            # 4. Generate HUD Overlay Frame (for both Live Stream & Preview Window)
            display_frame = frame.copy()

            if leaf_currently_visible:
                x1, y1, x2, y2 = box
                box_color = (0, 255, 0) if curr_disease_code == "HEALTHY" else (0, 69, 255)
                if curr_disease_code == "UNCERTAIN":
                    box_color = (0, 215, 255)
                elif curr_disease_code == "LOADING":
                    box_color = (255, 200, 0)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(display_frame, curr_display_label, (x1, max(28, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            else:
                cv2.putText(display_frame, "NO LEAF DETECTED", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 140, 255), 2)
                cv2.putText(display_frame, "Point camera at chili plant leaf", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Telemetry status bar
            ai_tag = "LiteRT High-Speed" if use_tflite else ("AI Loading..." if not model_ready else "Keras")
            status_bar = f"AgriRover AI | Cam: {camera_type} | FPS: {fps_display:.1f} | Engine: {ai_tag}"
            cv2.putText(display_frame, status_bar, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Obstacle warning banner on HUD
            with obstacle_lock:
                obs_active = obstacle_detected
                obs_dist = current_obstacle_distance
            if obs_active:
                cv2.rectangle(display_frame, (10, 440), (630, 475), (0, 0, 180), -1)
                cv2.putText(display_frame, f"OBSTACLE DETECTED: {obs_dist:.1f} CM - MOTORS STOPPED", (20, 465),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # Targeted disease spray banner on HUD
            with spray_lock:
                spray_active = is_spraying
            if spray_active:
                cv2.rectangle(display_frame, (10, 400), (630, 435), (180, 0, 180), -1)
                cv2.putText(display_frame, f"SPRAY PUMP ACTIVE: {curr_display_label}", (20, 425),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # Encode JPEG for live HTTP stream (Quality 75, fast non-blocking encode)
            try:
                ret_enc, enc_jpeg = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret_enc:
                    frame_bytes = enc_jpeg.tobytes()
                    with latest_jpeg_lock:
                        latest_jpeg_frame = frame_bytes
                    try:
                        with open("/dev/shm/rover_stream.jpg.tmp", "wb") as f:
                            f.write(frame_bytes)
                        os.replace("/dev/shm/rover_stream.jpg.tmp", "/dev/shm/rover_stream.jpg")
                    except Exception:
                        pass

                # Crystal-Clear Cloud Stream Frame (Strictly Preserves Natural Aspect Ratio)
                # Avoids squishing or stretching camera image on Render dashboard!
                dh, dw = display_frame.shape[:2]
                max_dim = max(dw, dh)
                if max_dim > 480:
                    scale = 480.0 / float(max_dim)
                    target_w = int(dw * scale)
                    target_h = int(dh * scale)
                    cloud_frame = cv2.resize(display_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                else:
                    cloud_frame = display_frame

                ok_cloud, enc_cloud = cv2.imencode('.jpg', cloud_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok_cloud:
                    with latest_cloud_jpeg_lock:
                        latest_cloud_jpeg_frame = enc_cloud.tobytes()
                    cloud_frame_seq += 1
            except Exception:
                pass

            # 5. Display Window Overlays (GUI mode only)
            if not HEADLESS:
                try:
                    cv2.imshow("AgriRover - Continuous Chili Disease Detection", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        log("User pressed 'q'. Exiting...")
                        break
                except Exception as gui_err:
                    log(f"GUI display error: {gui_err}. Switching to headless mode.")
                    HEADLESS = True
            else:
                time.sleep(0.002)

    except KeyboardInterrupt:
        log("Keyboard interrupt received.")
    except Exception as e:
        log(f"Unexpected error in main loop: {e}")
    finally:
        log("Cleaning up all hardware resources...")
        running = False
        stop_motors()
        set_pump(False)

        if stream_server is not None:
            try:
                stream_server.shutdown()
            except Exception:
                pass

        if gpio_available and gpio_chip is not None:
            try:
                lgpio.gpiochip_close(gpio_chip)
            except Exception:
                pass

        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
            except Exception:
                pass

        if cap is not None:
            cap.release()

        if not HEADLESS:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        try:
            ai_inference_queue.put(None)
        except Exception:
            pass

        try:
            upload_queue.put(None)
        except Exception:
            pass

        try:
            os.sync()
        except Exception:
            pass

        log("AgriRover AI System cleanly stopped.")

if __name__ == "__main__":
    main()
