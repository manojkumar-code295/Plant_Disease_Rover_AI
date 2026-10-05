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

# AI Confidence & Classification Thresholds
# If confidence < CONFIDENCE_THRESHOLD, reported as UNCERTAIN
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.60))

# Leaf Filtering Thresholds
MIN_LEAF_AREA = int(os.environ.get("MIN_LEAF_AREA", 4000))           # Min pixels to qualify as a leaf
MAX_LEAF_AREA_RATIO = float(os.environ.get("MAX_LEAF_AREA_RATIO", 0.65)) # Max 65% of frame (rejects room/ceiling)
MIN_LEAF_DENSITY = float(os.environ.get("MIN_LEAF_DENSITY", 0.38))   # At least 38% of bounding box must be foliage
MAX_SKIN_RATIO = float(os.environ.get("MAX_SKIN_RATIO", 0.30))       # Skin tone ratio threshold for rejecting full face/hands
MIN_TEXTURE_VAR = float(os.environ.get("MIN_TEXTURE_VAR", 50.0))     # Laplacian variance to reject flat walls and blurry reflections

# Camera & Performance Configuration
# Mounting orientation: "90_CCW" (default for upright rover chassis mount), "90_CW", "180", "NONE"
CAMERA_ROTATION = os.environ.get("CAMERA_ROTATION", "90_CCW")
PREDICT_EVERY_N_FRAMES = int(os.environ.get("PREDICT_EVERY_N_FRAMES", 2))
UPLOAD_INTERVAL = float(os.environ.get("UPLOAD_INTERVAL", 4.0))      # Minimum seconds between cloud uploads

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

def stop_motors():
    gpio_write(LEFT_IN1, 0)
    gpio_write(LEFT_IN2, 0)
    gpio_write(LEFT_EN, 0)
    gpio_write(RIGHT_IN1, 0)
    gpio_write(RIGHT_IN2, 0)
    gpio_write(RIGHT_EN, 0)
    log("GPIO OUTPUT -> All motor pins driven LOW (0V - Stopped)")

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

latest_jpeg_frame = None
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
    direction = str(direction).lower().strip()
    is_pump = bool(pump)
    set_pump(is_pump)
    shared_telemetry["pump"] = is_pump

    if direction in ("stop", "emergency_stop", "stopped", ""):
        shared_telemetry["direction"] = "Stopped"
        stop_motors()
        return

    shared_telemetry["direction"] = direction.capitalize()
    is_moving = speed > 0
    in1, in2, in3, in4 = 0, 0, 0, 0
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

    gpio_write(LEFT_IN1, in1)
    gpio_write(LEFT_IN2, in2)
    gpio_write(RIGHT_IN1, in3)
    gpio_write(RIGHT_IN2, in4)
    gpio_write(LEFT_EN, is_moving)
    gpio_write(RIGHT_EN, is_moving)
    log(f"GPIO OUTPUT -> IN1(Pin 11)={in1} | IN2(Pin 13)={in2} | IN3(Pin 15)={in3} | IN4(Pin 16)={in4} | ENA(Pin 32)={int(is_moving)} | ENB(Pin 33)={int(is_moving)}")

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
# LOAD AI MODEL (High-Speed TFLite with Keras Fallback)
# ============================================================

use_tflite = False
tflite_interpreter = None
keras_model = None

if os.path.exists(TFLITE_MODEL_PATH):
    try:
        log("Importing AI acceleration engine (TensorFlow Lite)...")
        import tensorflow as tf
        log(f"Loading high-speed TFLite model: {TFLITE_MODEL_PATH}")
        tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        tflite_interpreter.allocate_tensors()
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        use_tflite = True
        log("TFLite Model loaded successfully with XNNPACK acceleration!")
    except Exception as e:
        log(f"TFLite load failed ({e}), falling back to Keras...")
        use_tflite = False

if not use_tflite:
    if os.path.exists(KERAS_MODEL_PATH):
        import tensorflow as tf
        log(f"Loading full Keras model: {KERAS_MODEL_PATH}")
        keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
        log("Keras model loaded successfully.")
    else:
        log("ERROR: No AI model file found!")
        sys.exit(1)

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

def detect_leaf(frame_bgr):
    """
    Detects real plant leaves and strictly rejects faces, hands, walls, ceilings, and room clutter.
    Scores candidates by botanical vegetation metrics (foliage density, Excess Green, solidity),
    preventing large dark background objects (backpacks, table edges) from taking precedence.
    Returns:
        crop_bgr, (x1, y1, x2, y2), "LEAF_DETECTED" if a real leaf is present.
        None, None, "NO_LEAF" if no leaf is in view.
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
    # Real plant leaves have active chlorophyll where Green strictly dominates Red and Blue.
    # Eliminates beige walls, wooden tables, grey computer cases, vent grilles, and blue reflections.
    foliage_bool = (hsv[:,:,0] >= 28) & (hsv[:,:,0] <= 90) & (hsv[:,:,1] >= 45) & \
                   (hsv[:,:,2] >= 35) & (hsv[:,:,2] <= 245) & \
                   (g_full > r_full + 6.0) & (g_full > b_full + 8.0) & \
                   (2.0 * g_full - r_full - b_full >= 18.0)
    foliage_mask = np.uint8(foliage_bool) * 255

    # 2. Human Skin Tone Mask: catches faces, hands, arms holding leaves
    skin1 = cv2.inRange(hsv, np.array([0, 25, 45], dtype=np.uint8),
                             np.array([22, 175, 255], dtype=np.uint8))
    skin2 = cv2.inRange(hsv, np.array([165, 25, 45], dtype=np.uint8),
                             np.array([180, 175, 255], dtype=np.uint8))
    skin_mask = cv2.bitwise_or(skin1, skin2)

    # Subtract skin pixels from foliage so leaf contour is isolated even if held by hand
    foliage_clean = cv2.bitwise_and(foliage_mask, cv2.bitwise_not(skin_mask))

    # Morphological bridging to connect leaf blade segments
    k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    foliage_clean = cv2.morphologyEx(foliage_clean, cv2.MORPH_CLOSE, k_bridge)

    contours, _ = cv2.findContours(foliage_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    best_score = -1.0

    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            # Area must be between minimum leaf size (4000px) and maximum leaf ratio
            if area < MIN_LEAF_AREA or area > max_leaf_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)

            # Real leaves close to camera must have substantial width and height
            if bw < 70 or bh < 70:
                continue

            # Reject contours spanning almost full width or height (room walls, ceiling, desk width)
            if bw >= int(w * 0.75) or bh >= int(h * 0.75):
                continue

            aspect = bw / float(bh)
            if aspect < 0.25 or aspect > 4.0:
                continue

            box_area = float(bw * bh)
            crop_foliage = foliage_clean[by:by+bh, bx:bx+bw]
            foliage_px = cv2.countNonZero(crop_foliage)
            if foliage_px < 2800:
                continue
            leaf_ratio = foliage_px / box_area

            # Contour solidity check: real leaves have organic connected shapes; room clutter is disjoint
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / float(hull_area) if hull_area > 0 else 0
            if solidity < 0.48:
                continue

            # Foliage density inside bounding box
            if leaf_ratio < MIN_LEAF_DENSITY:
                continue

            # Skin check: if the bounding box is predominantly skin (> 30%), reject as person/hand
            crop_skin = skin_mask[by:by+bh, bx:bx+bw]
            skin_px = cv2.countNonZero(crop_skin)
            skin_ratio = skin_px / box_area
            if skin_ratio > MAX_SKIN_RATIO or skin_px > foliage_px:
                continue

            # Texture check: Real leaves have veins and margins; flat walls and blurry reflections have low variance
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

            if mean_exg < 20.0:
                continue

            # Botanical score: prioritizes true green foliage density, Excess Green, and organic solidity
            score = foliage_px * (mean_exg + 5.0) * solidity

            if score > best_score:
                best_score = score
                pad = 14
                cx1 = max(0, bx - pad); cy1 = max(0, by - pad)
                cx2 = min(w, bx + bw + pad); cy2 = min(h, by + bh + pad)
                best_candidate = (cx1, cy1, cx2, cy2)

    if best_candidate is not None:
        x1, y1, x2, y2 = best_candidate
        crop = frame_bgr[y1:y2, x1:x2].copy()
        if crop.size > 0:
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

    try:
        # Resize to standard model input dimensions (IMAGE_SIZE, IMAGE_SIZE)
        resized = cv2.resize(crop_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        # Convert BGR to RGB (model expects RGB)
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = np.expand_dims(img, axis=0)

        if use_tflite:
            tflite_interpreter.set_tensor(input_details[0]['index'], img)
            tflite_interpreter.invoke()
            preds = tflite_interpreter.get_tensor(output_details[0]['index'])[0]
        else:
            preds = keras_model.predict(img, verbose=0)[0]

        best_idx = int(np.argmax(preds))
        conf = float(preds[best_idx])
        raw_class = class_names[best_idx]

        # Handle confidence threshold
        if conf < CONFIDENCE_THRESHOLD:
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
                r = upload_session.post(UPDATE_URL, files=files, data=data, headers=headers, timeout=12)
                if r.ok:
                    log(f"CLOUD SYNC -> Render updated: {display_name}")
                else:
                    log(f"CLOUD SYNC -> HTTP {r.status_code}")
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

last_command_id = -1
last_local_command_id = -1

def rover_telemetry_worker():
    """Background thread polling commands from both local Flask server and Render cloud."""
    global last_command_id, last_local_command_id
    poll_session = requests.Session()
    last_hb = 0.0

    while running:
        now = time.time()
        # Send Heartbeat with rich telemetry payload every 4 seconds to both Render and Local server
        if now - last_hb >= 4.0:
            headers = {"Content-Type": "application/json"}
            if RENDER_API_TOKEN:
                headers["Authorization"] = f"Bearer {RENDER_API_TOKEN}"
            try:
                poll_session.post(HEARTBEAT_URL, json=shared_telemetry, headers=headers, timeout=4)
            except Exception:
                pass
            try:
                poll_session.post("http://127.0.0.1:5000/api/rover/heartbeat", json=shared_telemetry, headers={"Content-Type": "application/json"}, timeout=1)
            except Exception:
                pass
            last_hb = now

        # 1. Poll Local Server (Instant zero-latency response on local Wi-Fi / Pi screen)
        try:
            r_loc = poll_session.get("http://127.0.0.1:5000/api/rover/command", timeout=0.25)
            if r_loc.ok:
                data_loc = r_loc.json()
                cid_loc = data_loc.get("command_id", 0)
                if cid_loc != last_local_command_id and cid_loc > 0:
                    direction = data_loc.get("direction", "Stopped")
                    speed = int(data_loc.get("speed", 50))
                    pump = bool(data_loc.get("pump", False))
                    log(f"ROVER COMMAND (Local Web) -> {direction} | Speed: {speed}% | Pump: {'ON' if pump else 'OFF'}")
                    apply_rover_command(direction, speed, pump)
                    last_local_command_id = cid_loc
        except Exception:
            pass

        # 2. Poll Render Cloud (Remote web dashboard)
        try:
            headers = {}
            if RENDER_API_TOKEN:
                headers["Authorization"] = f"Bearer {RENDER_API_TOKEN}"
            r = poll_session.get(COMMAND_URL, headers=headers, timeout=4)
            if r.ok:
                data = r.json()
                cid = data.get("command_id", 0)
                if cid != last_command_id and cid > 0:
                    direction = data.get("direction", "Stopped")
                    speed = int(data.get("speed", 50))
                    pump = bool(data.get("pump", False))
                    log(f"ROVER COMMAND (Cloud Render) -> {direction} | Speed: {speed}% | Pump: {'ON' if pump else 'OFF'}")
                    apply_rover_command(direction, speed, pump)
                    last_command_id = cid
        except Exception:
            pass

        time.sleep(0.35)

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    global running, HEADLESS, latest_jpeg_frame

    # Start background workers
    upload_thread = threading.Thread(target=upload_worker, daemon=True)
    upload_thread.start()

    telemetry_thread = threading.Thread(target=rover_telemetry_worker, daemon=True)
    telemetry_thread.start()

    # Camera initialization
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
        time.sleep(1.0)
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
    last_upload_time = 0.0
    curr_disease_code = "NO_LEAF"
    curr_display_label = "NO LEAF DETECTED"
    curr_conf = 0.0
    curr_box = None
    curr_mode = "NO_LEAF"
    consec_class = None
    consec_count = 0

    stream_server = start_streaming_server(STREAM_PORT)

    log(f"AgriRover AI System Active. Headless mode: {HEADLESS}")
    if not HEADLESS:
        log("Press 'q' in the preview window to exit.")

    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0.0

    try:
        while running:
            # 1. Grab Raw Frame
            if camera_type == "PICAMERA2":
                raw_rgb = picam2.capture_array()
                # Picamera2 returns RGB888. Convert to OpenCV BGR standard
                frame = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
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

            # 2. Strict Leaf Identification (Rejects faces, hands, walls)
            crop, box, mode = detect_leaf(frame)
            curr_box = box
            curr_mode = mode

            # 3. Plant Disease Classification (Only executed when a real leaf is present)
            if mode == "LEAF_DETECTED" and crop is not None and crop.size > 0:
                if frame_count % PREDICT_EVERY_N_FRAMES == 0:
                    disease_code, display_label, conf, _ = predict_crop(crop)
                    curr_disease_code = disease_code
                    curr_display_label = display_label
                    curr_conf = conf

                    if curr_disease_code == consec_class:
                        consec_count += 1
                    else:
                        consec_class = curr_disease_code
                        consec_count = 1

                    shared_telemetry["last_detection"] = curr_display_label
                    shared_telemetry["last_confidence"] = round(curr_conf * 100.0 if curr_conf <= 1.0 else curr_conf, 1)
                    shared_telemetry["ai_status"] = f"Classified: {curr_display_label}"

                    # Only upload CONFIRMED classifications (HEALTHY, Bacterial_Spot, Cercospora_Leaf_Spot, Curl_Virus)
                    # Require at least 2 consecutive confirmations to prevent transient flickers
                    # Never upload UNCERTAIN or NO_LEAF guesses to the web dashboard
                    valid_disease_classes = ("HEALTHY", "Bacterial_Spot", "Cercospora_Leaf_Spot", "Curl_Virus")
                    if (curr_disease_code in valid_disease_classes) and (conf >= CONFIDENCE_THRESHOLD) and (consec_count >= 2) and (now - last_upload_time >= UPLOAD_INTERVAL):
                        snap_path = os.path.join(SAVE_FOLDER, f"leaf_{int(now)}.jpg")
                        cv2.imwrite(snap_path, crop)
                        try:
                            os.sync()
                        except Exception:
                            pass

                        try:
                            if upload_queue.full():
                                upload_queue.get_nowait()
                            upload_queue.put_nowait((crop.copy(), curr_disease_code, curr_display_label, curr_conf))
                            last_upload_time = now
                        except Exception:
                            pass
            else:
                consec_class = None
                consec_count = 0
                curr_disease_code = "NO_LEAF"
                curr_display_label = "NO LEAF DETECTED"
                curr_conf = 0.0
                shared_telemetry["last_detection"] = "NO LEAF DETECTED"
                shared_telemetry["last_confidence"] = 0.0
                shared_telemetry["ai_status"] = "Active Scanning (No Leaf in View)"

            # 4. Generate HUD Overlay Frame (for both Live Stream & Preview Window)
            display_frame = frame.copy()

            if curr_mode == "LEAF_DETECTED" and curr_box is not None:
                x1, y1, x2, y2 = curr_box
                box_color = (0, 255, 0) if curr_disease_code == "HEALTHY" else (0, 69, 255)
                if curr_disease_code == "UNCERTAIN":
                    box_color = (0, 215, 255)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(display_frame, curr_display_label, (x1, max(28, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            else:
                # Clear, unambiguous state when no leaf is in view
                cv2.putText(display_frame, "NO LEAF DETECTED", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 140, 255), 2)
                cv2.putText(display_frame, "Point camera at chili plant leaf", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Telemetry status bar
            status_bar = f"AgriRover AI | Camera: {camera_type} | FPS: {fps_display:.1f} | State: {curr_mode}"
            cv2.putText(display_frame, status_bar, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Encode JPEG for live HTTP stream
            try:
                ret_enc, enc_jpeg = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret_enc:
                    with latest_jpeg_lock:
                        latest_jpeg_frame = enc_jpeg.tobytes()
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
                time.sleep(0.005)

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
