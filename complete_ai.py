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
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.72))
CONFIDENCE_MARGIN = float(os.environ.get("CONFIDENCE_MARGIN", 0.25))

# Robust Leaf Filtering Thresholds (Botanical Quality Gate for Chili Leaves)
MIN_LEAF_AREA = int(os.environ.get("MIN_LEAF_AREA", 1200))           # Min pixels to qualify as a leaf
MAX_LEAF_AREA_RATIO = float(os.environ.get("MAX_LEAF_AREA_RATIO", 0.75)) # Max 75% of frame (rejects whole room)
MIN_LEAF_DENSITY = float(os.environ.get("MIN_LEAF_DENSITY", 0.20))   # Foliage density inside bounding box
MAX_SKIN_RATIO = float(os.environ.get("MAX_SKIN_RATIO", 0.40))       # Skin tone ratio threshold for rejecting full face
MIN_TEXTURE_VAR = float(os.environ.get("MIN_TEXTURE_VAR", 10.0))     # Laplacian variance to reject flat blank walls
MIN_EXG_THRESHOLD = float(os.environ.get("MIN_EXG_THRESHOLD", 8.0))  # Excess Green Index (chlorophyll presence)
MIN_SATURATION = float(os.environ.get("MIN_SATURATION", 25.0))       # Rejects gray/off-white rooms and furniture

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
TRIG_PIN  = 24  # Physical Pin 18 (HC-SR04 Trigger)
ECHO_PIN  = 25  # Physical Pin 22 (HC-SR04 Echo)

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
        pins = [LEFT_IN1, LEFT_IN2, LEFT_EN, RIGHT_IN1, RIGHT_IN2, RIGHT_EN, PUMP_PIN]
        for pin in pins:
            try:
                lgpio.gpio_free(gpio_chip, pin)
            except Exception:
                pass
            lgpio.gpio_claim_output(gpio_chip, pin, 0)

        gpio_available = True
        log("GPIO Controller initialized. All motors and pump safely initialized in OFF state.")
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

def stop_motors():
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
    gpio_write(PUMP_PIN, state)

# Ensure motors are completely stopped right after init
stop_motors()
set_pump(False)

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
    global INVERT_RIGHT_MOTOR, INVERT_LEFT_MOTOR
    direction = str(direction).lower().strip()
    is_pump = bool(pump)
    set_pump(is_pump)
    shared_telemetry["pump"] = is_pump

    if direction in ("stop", "emergency_stop", "stopped", ""):
        shared_telemetry["direction"] = "Stopped"
        shared_telemetry["engine_status"] = "Standby"
        stop_motors()
        return

    # Dedicated START Command: Arms the Engine / Mission System
    # Does NOT actuate directional wheels until a directional command (FWD/REV/LFT/RGT) is given!
    if direction == "start":
        shared_telemetry["direction"] = "Engine Armed"
        shared_telemetry["engine_status"] = "Running"
        stop_motors()
        log("ENGINE ARMED / SYSTEM READY -> Drive system armed, standing by for directional commands.")
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
    elif direction == "left":
        in1, in2 = 0, 1
        in3, in4 = 1, 0
    elif direction == "right":
        in1, in2 = 1, 0
        in3, in4 = 0, 1

    # Motor Polarity Inversion (Fixes differential drive mirror mount spinning)
    if INVERT_LEFT_MOTOR:
        in1, in2 = in2, in1
    if INVERT_RIGHT_MOTOR:
        in3, in4 = in4, in3

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
        time.sleep(0.01)
        gpio_write(LEFT_EN, 1)
        gpio_write(RIGHT_EN, 1)
        gpio_pwm(LEFT_IN1, speed if in1 else 0)
        gpio_pwm(LEFT_IN2, speed if in2 else 0)
        gpio_pwm(RIGHT_IN1, speed if in3 else 0)
        gpio_pwm(RIGHT_IN2, speed if in4 else 0)
    else:
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
    Botanical Quality Gate: Verifies that a candidate image is a genuine plant leaf.
    Strictly rejects desk surfaces, shadows, walls, curtains, carpets, furniture, and hands.
    Returns: (True, "OK") if genuine foliage; (False, reason) otherwise.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False, "Empty image"

    h, w = crop_bgr.shape[:2]
    if w < 75 or h < 75 or (w * h) < 6000:
        return False, f"Candidate too small ({w}x{h})"

    # Botanical aspect ratio: Chili leaves are elliptical/lanceolate (0.35 <= w/h <= 2.6)
    aspect = w / float(h)
    if aspect < 0.35 or aspect > 2.6:
        return False, f"Non-botanical aspect ratio ({aspect:.2f})"

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    b = crop_bgr[:, :, 0].astype(np.float32)
    g = crop_bgr[:, :, 1].astype(np.float32)
    r = crop_bgr[:, :, 2].astype(np.float32)

    # 1. Botanical chlorophyll foliage mask:
    # Requires genuine plant green spectrum OR high-saturation chlorotic yellow
    green_mask = (hsv[:,:,0] >= 26) & (hsv[:,:,0] <= 86) & \
                 (hsv[:,:,1] >= 30) & (hsv[:,:,2] >= 35) & (hsv[:,:,2] <= 250) & \
                 (g > r + 3.0) & (g > b + 5.0) & (2.0 * g - r - b >= 8.0)

    yellow_mask = (hsv[:,:,0] >= 16) & (hsv[:,:,0] < 28) & \
                  (hsv[:,:,1] >= 35) & (hsv[:,:,2] >= 40) & (hsv[:,:,2] <= 250) & \
                  (g > b + 10.0) & (r > b + 10.0)

    foliage_mask = green_mask | yellow_mask
    foliage_px = int(np.sum(foliage_mask))
    total_px = float(h * w)
    foliage_ratio = foliage_px / total_px

    if foliage_px < 500:
        return False, f"Insufficient chlorophyll pixels ({foliage_px} < 500)"
    if foliage_ratio < 0.15:
        return False, f"Low foliage ratio ({foliage_ratio*100:.1f}% < 15%)"

    # 2. Human Skin Tone Rejection
    skin1 = cv2.inRange(hsv, np.array([0, 25, 45], dtype=np.uint8),
                             np.array([22, 175, 255], dtype=np.uint8))
    skin2 = cv2.inRange(hsv, np.array([165, 25, 45], dtype=np.uint8),
                             np.array([180, 175, 255], dtype=np.uint8))
    skin_mask = cv2.bitwise_or(skin1, skin2)
    skin_px = int(np.sum(skin_mask > 0))
    if skin_px > foliage_px * 2.0 or (skin_px / total_px) > MAX_SKIN_RATIO:
        return False, "Hand/Skin detected"

    # 3. Saturation & Excess Green on Foliage
    sat_foliage = float(np.mean(hsv[:,:,1][foliage_mask]))
    if sat_foliage < 22.0:
        return False, f"Desaturated foliage ({sat_foliage:.1f} < 22.0)"

    exg = 2.0 * g - r - b
    mean_exg = float(np.mean(exg[foliage_mask]))
    if mean_exg < 6.0:
        return False, f"Low chlorophyll ExG ({mean_exg:.1f} < 6.0)"

    # 4. Texture & Venation Check
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 8.0:
        return False, f"Flat texture ({lap_var:.1f} < 8.0)"

    return True, "VALID_LEAF"

def detect_leaf(frame_bgr):
    """
    Detects real plant leaves (both healthy and diseased with spots/yellowing/curling)
    and strictly rejects faces, hands, walls, ceilings, and room clutter.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None, None, "NO_LEAF"

    h, w = frame_bgr.shape[:2]
    total_area = float(h * w)
    max_leaf_area = int(total_area * MAX_LEAF_AREA_RATIO)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    b_full = frame_bgr[:, :, 0].astype(np.float32)
    g_full = frame_bgr[:, :, 1].astype(np.float32)
    r_full = frame_bgr[:, :, 2].astype(np.float32)

    # 1. Botanical Foliage Mask with Chlorophyll Discrimination:
    green_bool = (hsv[:,:,0] >= 26) & (hsv[:,:,0] <= 86) & \
                 (hsv[:,:,1] >= 42) & (hsv[:,:,2] >= 42) & (hsv[:,:,2] <= 245) & \
                 (g_full > r_full + 6.0) & (g_full > b_full + 8.0) & \
                 (2.0 * g_full - r_full - b_full >= 20.0)

    yellow_bool = (hsv[:,:,0] >= 18) & (hsv[:,:,0] < 26) & \
                  (hsv[:,:,1] >= 50) & (hsv[:,:,2] >= 45) & (hsv[:,:,2] <= 245) & \
                  (g_full > b_full + 15.0) & (r_full > b_full + 15.0)

    foliage_mask = np.uint8(green_bool | yellow_bool) * 255

    # 2. Human Skin Tone Mask: catches faces, hands, arms holding leaves
    skin1 = cv2.inRange(hsv, np.array([0, 25, 45], dtype=np.uint8),
                             np.array([22, 175, 255], dtype=np.uint8))
    skin2 = cv2.inRange(hsv, np.array([165, 25, 45], dtype=np.uint8),
                             np.array([180, 175, 255], dtype=np.uint8))
    skin_mask = cv2.bitwise_or(skin1, skin2)

    # Subtract skin pixels from foliage so leaf contour is isolated even if held by hand
    foliage_clean = cv2.bitwise_and(foliage_mask, cv2.bitwise_not(skin_mask))

    # Morphological bridging to connect leaf blade segments
    k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    foliage_clean = cv2.morphologyEx(foliage_clean, cv2.MORPH_CLOSE, k_bridge)

    contours, _ = cv2.findContours(foliage_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    best_score = 0.0

    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            # Area must be between minimum leaf size and maximum leaf ratio
            if area < MIN_LEAF_AREA or area > max_leaf_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 55 or bh < 55:
                continue

            # Reject contours spanning almost full width or height (room walls, ceiling, desk width)
            if bw >= int(w * 0.85) or bh >= int(h * 0.85):
                continue

            aspect = bw / float(bh)
            if aspect < 0.35 or aspect > 2.6:
                continue

            box_area = float(bw * bh)
            crop_foliage = foliage_clean[by:by+bh, bx:bx+bw]
            foliage_px = cv2.countNonZero(crop_foliage)
            if foliage_px < 1000:
                continue
            leaf_ratio = foliage_px / box_area

            # Contour solidity check: real leaves have organic connected shapes; room clutter is disjoint
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / float(hull_area) if hull_area > 0 else 0
            if solidity < 0.42:
                continue

            # Foliage density inside bounding box
            if leaf_ratio < MIN_LEAF_DENSITY:
                continue

            # Skin check: if the bounding box is predominantly skin, reject
            crop_skin = skin_mask[by:by+bh, bx:bx+bw]
            skin_px = cv2.countNonZero(crop_skin)
            skin_ratio = skin_px / box_area
            if skin_ratio > MAX_SKIN_RATIO or skin_px > foliage_px:
                continue

            # Texture check: Real leaves have veins and margins
            leaf_bgr = frame_bgr[by:by+bh, bx:bx+bw]
            if leaf_bgr.size == 0:
                continue

            gray = cv2.cvtColor(leaf_bgr, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < MIN_TEXTURE_VAR:
                continue

            # Botanical Vegetation Index: Excess Green (2G - R - B) on the foliage pixels
            b_ch = leaf_bgr[:, :, 0].astype(np.float32)
            g_ch = leaf_bgr[:, :, 1].astype(np.float32)
            r_ch = leaf_bgr[:, :, 2].astype(np.float32)
            exg = 2.0 * g_ch - r_ch - b_ch
            foliage_mask_crop = crop_foliage > 0
            mean_exg = float(np.mean(exg[foliage_mask_crop])) if foliage_px > 0 else 0.0

            if mean_exg < MIN_EXG_THRESHOLD:
                continue

            # Botanical score: prioritizes true green foliage density, Excess Green, and organic solidity
            score = foliage_px * (mean_exg + 5.0) * solidity

            if score >= 25000.0 and score > best_score:
                best_score = score
                pad = 12
                cx1 = max(0, bx - pad); cy1 = max(0, by - pad)
                cx2 = min(w, bx + bw + pad); cy2 = min(h, by + bh + pad)
                best_candidate = (cx1, cy1, cx2, cy2)

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
        # Resize to standard model input dimensions (IMAGE_SIZE, IMAGE_SIZE)
        resized = cv2.resize(crop_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
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

            # Continuous disease upload: immediately on change, or every 2s for continuous updates
            valid_disease_classes = ("HEALTHY", "Bacterial_Spot", "Cercospora_Leaf_Spot", "Curl_Virus")
            is_valid_disease = (disease_code in valid_disease_classes) and (conf >= CONFIDENCE_THRESHOLD)

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
last_local_command_id = -1

def local_command_worker():
    """Responsive worker for local web dashboard commands (sub-120ms latency, low CPU)."""
    global last_local_command_id
    session = requests.Session()
    while running:
        try:
            r = session.get("http://127.0.0.1:5000/api/rover/command", timeout=0.25)
            if r.ok:
                data = r.json()
                cid = data.get("command_id", 0)
                if cid != last_local_command_id and cid > 0:
                    direction = data.get("direction", "Stopped")
                    speed = int(data.get("speed", 50))
                    pump = bool(data.get("pump", False))
                    if "invert_right_motor" in data:
                        INVERT_RIGHT_MOTOR = bool(data["invert_right_motor"])
                    if "invert_left_motor" in data:
                        INVERT_LEFT_MOTOR = bool(data["invert_left_motor"])
                    log(f"ROVER COMMAND (Local Web) -> {direction} | Speed: {speed}% | Pump: {'ON' if pump else 'OFF'}")
                    apply_rover_command(direction, speed, pump)
                    last_local_command_id = cid
        except Exception:
            pass
        time.sleep(0.12)

def cloud_command_worker():
    """Dedicated cloud command polling thread for Render / Android App (1.2s cadence)."""
    global last_command_id, INVERT_RIGHT_MOTOR, INVERT_LEFT_MOTOR
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
                if cid != last_command_id and cid > 0:
                    direction = data.get("direction", "Stopped")
                    speed = int(data.get("speed", 50))
                    pump = bool(data.get("pump", False))
                    if "invert_right_motor" in data:
                        INVERT_RIGHT_MOTOR = bool(data["invert_right_motor"])
                    if "invert_left_motor" in data:
                        INVERT_LEFT_MOTOR = bool(data["invert_left_motor"])
                    log(f"ROVER COMMAND (Cloud Render) -> {direction} | Speed: {speed}% | Pump: {'ON' if pump else 'OFF'}")
                    apply_rover_command(direction, speed, pump)
                    last_command_id = cid
        except Exception:
            pass
        time.sleep(0.18)

def cloud_frame_streamer_worker():
    """Streams live camera frames to Render so the remote dashboard & phone see live video."""
    session = requests.Session()
    headers = {"Content-Type": "image/jpeg"}
    if RENDER_API_TOKEN:
        headers["Authorization"] = f"Bearer {RENDER_API_TOKEN}"
    stream_url = RENDER_URL + "/api/camera/stream_frame"

    while running:
        try:
            with latest_jpeg_lock:
                frame = latest_jpeg_frame
            if frame is not None:
                session.post(stream_url, data=frame, headers=headers, timeout=1.2)
            time.sleep(0.35)
        except Exception:
            time.sleep(0.8)


def local_telemetry_worker():
    """High-speed local telemetry sender (updates local web dashboard every 0.5s with zero WAN lag)."""
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
        time.sleep(0.5)

def cloud_telemetry_worker():
    """Dedicated cloud telemetry heartbeat sender to Render cloud (every 1.0s)."""
    session = requests.Session()
    headers = {"Content-Type": "application/json"}
    if RENDER_API_TOKEN:
        headers["Authorization"] = f"Bearer {RENDER_API_TOKEN}"

    while running:
        try:
            session.post(HEARTBEAT_URL, json=shared_telemetry, headers=headers, timeout=1.2)
        except Exception:
            pass
        time.sleep(1.0)

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    global running, HEADLESS, latest_jpeg_frame

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

    frame_stream_thread = threading.Thread(target=cloud_frame_streamer_worker, daemon=True, name="CloudFrameStreamer")
    frame_stream_thread.start()

    local_telemetry_thread = threading.Thread(target=local_telemetry_worker, daemon=True, name="LocalTelemetryWorker")
    local_telemetry_thread.start()

    cloud_telemetry_thread = threading.Thread(target=cloud_telemetry_worker, daemon=True, name="CloudTelemetryWorker")
    cloud_telemetry_thread.start()

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
        if not cap.isOpened():
            log("ERROR: Could not open camera via Picamera2 or VideoCapture(0).")
            sys.exit(1)
        camera_type = "OPENCV_V4L2"
        log("Camera started via OpenCV V4L2 fallback.")

    frame_count = 0
    curr_disease_code = "NO_LEAF"
    curr_display_label = "NO LEAF DETECTED"
    curr_conf = 0.0
    last_box = None
    last_mode = "NO_LEAF"

    log(f"AgriRover AI System Active. Headless mode: {HEADLESS}")
    if not HEADLESS:
        log("Press 'q' in the preview window to exit.")

    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0.0

    try:
        while running:
            # 1. Grab Raw Frame at high speed
            if camera_type == "PICAMERA2":
                raw_rgb = picam2.capture_array()
                frame = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

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
            if frame_count % PREDICT_EVERY_N_FRAMES == 0:
                crop, box, mode = detect_leaf(frame)
                last_box = box
                last_mode = mode
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
            
            # Smooth detection retention window (1.5s): prevents 1-frame transient blinks from wiping telemetry
            if leaf_currently_visible or (now - ai_last_seen < 1.5 and ai_disease_code not in ("NO_LEAF", "LOADING", "INITIALIZING")):
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

            # Encode JPEG for live HTTP stream
            try:
                ret_enc, enc_jpeg = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret_enc:
                    frame_bytes = enc_jpeg.tobytes()
                    with latest_jpeg_lock:
                        latest_jpeg_frame = frame_bytes
                    try:
                        with open("/dev/shm/rover_stream.jpg", "wb") as f:
                            f.write(frame_bytes)
                    except Exception:
                        pass
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
