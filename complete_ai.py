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
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.50))

# Leaf Filtering Thresholds
MIN_LEAF_AREA = int(os.environ.get("MIN_LEAF_AREA", 3500))          # Min pixels to qualify as a leaf
MAX_LEAF_AREA_RATIO = float(os.environ.get("MAX_LEAF_AREA_RATIO", 0.70)) # Max 70% of frame (rejects room/ceiling)
MIN_LEAF_DENSITY = float(os.environ.get("MIN_LEAF_DENSITY", 0.22))  # At least 22% of bounding box must be foliage
MAX_SKIN_RATIO = float(os.environ.get("MAX_SKIN_RATIO", 0.14))      # Max 14% skin tone before rejecting as human
MIN_TEXTURE_VAR = float(os.environ.get("MIN_TEXTURE_VAR", 20.0))    # Laplacian variance to reject flat walls

# Camera & Performance Configuration
ROTATE_CAMERA_180 = os.environ.get("ROTATE_CAMERA_180", "1") == "1"
PREDICT_EVERY_N_FRAMES = int(os.environ.get("PREDICT_EVERY_N_FRAMES", 2))
UPLOAD_INTERVAL = float(os.environ.get("UPLOAD_INTERVAL", 2.0))     # Minimum seconds between cloud uploads

# Render Cloud Synchronization (Configurable via Environment)
RENDER_URL = os.environ.get("RENDER_URL", "https://precision-agriculture-smart-rover.onrender.com").rstrip("/")
RENDER_API_TOKEN = os.environ.get("RENDER_API_TOKEN", "")
UPDATE_URL = RENDER_URL + "/api/update-result"
HEARTBEAT_URL = RENDER_URL + "/api/rover/heartbeat"
COMMAND_URL = RENDER_URL + "/api/rover/command"

# Headless mode detection (systemd or no display)
HEADLESS = ("--headless" in sys.argv) or (not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"))

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

def set_pump(state):
    gpio_write(PUMP_PIN, state)

# Ensure motors are completely stopped right after init
stop_motors()
set_pump(False)

def apply_rover_command(direction, speed, pump):
    direction = str(direction).lower().strip()
    set_pump(bool(pump))

    if direction in ("stop", "emergency_stop", "stopped", ""):
        stop_motors()
        return

    is_moving = speed > 0
    if direction == "forward":
        gpio_write(LEFT_IN1, 1); gpio_write(LEFT_IN2, 0)
        gpio_write(RIGHT_IN1, 1); gpio_write(RIGHT_IN2, 0)
    elif direction in ("backward", "reverse"):
        gpio_write(LEFT_IN1, 0); gpio_write(LEFT_IN2, 1)
        gpio_write(RIGHT_IN1, 0); gpio_write(RIGHT_IN2, 1)
    elif direction == "left":
        gpio_write(LEFT_IN1, 0); gpio_write(LEFT_IN2, 1)
        gpio_write(RIGHT_IN1, 1); gpio_write(RIGHT_IN2, 0)
    elif direction == "right":
        gpio_write(LEFT_IN1, 1); gpio_write(LEFT_IN2, 0)
        gpio_write(RIGHT_IN1, 0); gpio_write(RIGHT_IN2, 1)

    gpio_write(LEFT_EN, is_moving)
    gpio_write(RIGHT_EN, is_moving)

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

def detect_leaf(frame_bgr):
    """
    Detects real plant leaves and strictly rejects faces, hands, walls, ceilings, and non-leaf objects.
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

    # 1. Plant Foliage Mask: covers healthy green, yellow-green, and chlorotic leaves
    # H: 22-95, S: 45-255, V: 25-250 (rejects pale mint walls and low-saturation backgrounds)
    leaf_mask = cv2.inRange(hsv, np.array([22, 45, 25], dtype=np.uint8),
                                 np.array([95, 255, 250], dtype=np.uint8))

    # 2. Human Skin Tone Mask: catches faces, hands, arms
    # H: 0-18 and 165-180 with natural skin saturation
    skin1 = cv2.inRange(hsv, np.array([0, 25, 45], dtype=np.uint8),
                             np.array([18, 180, 255], dtype=np.uint8))
    skin2 = cv2.inRange(hsv, np.array([165, 25, 45], dtype=np.uint8),
                             np.array([180, 180, 255], dtype=np.uint8))
    skin_mask = cv2.bitwise_or(skin1, skin2)

    # Morphological cleaning to isolate leaf clusters
    kernel = np.ones((5, 5), np.uint8)
    leaf_clean = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    leaf_clean = cv2.morphologyEx(leaf_clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(leaf_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    best_area = 0

    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            # Area must be between minimum leaf size and maximum leaf ratio
            if area < MIN_LEAF_AREA or area > max_leaf_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)

            # Reject contours spanning almost full width and height (room walls/ceiling)
            if bw >= int(w * 0.85) and bh >= int(h * 0.85):
                continue

            aspect = bw / float(bh)
            if aspect < 0.25 or aspect > 4.0:
                continue

            box_area = float(bw * bh)
            leaf_px = cv2.countNonZero(leaf_clean[by:by+bh, bx:bx+bw])
            skin_px = cv2.countNonZero(skin_mask[by:by+bh, bx:bx+bw])

            leaf_ratio = leaf_px / box_area
            skin_ratio = skin_px / box_area

            # REJECT IF:
            # 1. Leaf foliage is sparse inside bounding box (< 22%)
            # 2. Human skin is prominent (> 14% or skin exceeds half of foliage) -> Face or Hand!
            if leaf_ratio < MIN_LEAF_DENSITY or skin_ratio > MAX_SKIN_RATIO or skin_px > (leaf_px * 0.40):
                continue

            # 3. Texture check: Real leaves have veins, margins, spots; flat walls have near-zero Laplacian variance
            pad = 8
            cx1 = max(0, bx - pad); cy1 = max(0, by - pad)
            cx2 = min(w, bx + bw + pad); cy2 = min(h, by + bh + pad)
            test_crop = frame_bgr[cy1:cy2, cx1:cx2]
            if test_crop.size == 0:
                continue

            gray = cv2.cvtColor(test_crop, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < MIN_TEXTURE_VAR:
                continue

            # 4. Botanical Vegetation Index: Excess Green (2G - R - B) and Saturation
            # Real chili leaves have ExG >= 14 and Saturation >= 42
            # Flat grey/teal walls, ceilings, skin, and clothes have ExG < 14 or Saturation < 42
            b_ch = test_crop[:, :, 0].astype(np.float32)
            g_ch = test_crop[:, :, 1].astype(np.float32)
            r_ch = test_crop[:, :, 2].astype(np.float32)
            exg = 2.0 * g_ch - r_ch - b_ch
            mean_exg = float(np.mean(exg))

            crop_hsv = cv2.cvtColor(test_crop, cv2.COLOR_BGR2HSV)
            mean_sat = float(np.mean(crop_hsv[:, :, 1]))

            if mean_exg < 14.0 or mean_sat < 42.0:
                continue

            if area > best_area:
                best_area = area
                best_candidate = (cx1, cy1, cx2, cy2)

    if best_candidate is not None:
        x1, y1, x2, y2 = best_candidate
        crop = frame_bgr[y1:y2, x1:x2]
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
        img = cv2.resize(crop_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        # Convert BGR to RGB (model expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
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

                data = {
                    "disease": display_name,
                    "confidence": f"{confidence * 100:.2f}",
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

def rover_telemetry_worker():
    """Background thread polling commands from Render web dashboard."""
    global last_command_id
    poll_session = requests.Session()
    last_hb = 0.0

    while running:
        now = time.time()
        # Send Heartbeat every 5 seconds
        if now - last_hb >= 5.0:
            try:
                headers = {}
                if RENDER_API_TOKEN:
                    headers["Authorization"] = f"Bearer {RENDER_API_TOKEN}"
                poll_session.post(HEARTBEAT_URL, headers=headers, timeout=8)
                last_hb = now
            except Exception:
                pass

        # Poll movement / pump command every 0.4 seconds
        try:
            headers = {}
            if RENDER_API_TOKEN:
                headers["Authorization"] = f"Bearer {RENDER_API_TOKEN}"
            r = poll_session.get(COMMAND_URL, headers=headers, timeout=8)
            if r.ok:
                data = r.json()
                cid = data.get("command_id", 0)
                if cid != last_command_id:
                    direction = data.get("direction", "Stopped")
                    speed = int(data.get("speed", 50))
                    pump = bool(data.get("pump", False))
                    log(f"ROVER COMMAND -> {direction} | Speed: {speed}% | Pump: {'ON' if pump else 'OFF'}")
                    apply_rover_command(direction, speed, pump)
                    last_command_id = cid
        except Exception:
            pass

        time.sleep(0.4)

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    global running

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
            if ROTATE_CAMERA_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            frame_count += 1
            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

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

                    # Upload to Render if confidence threshold met and cooldown elapsed
                    if conf >= CONFIDENCE_THRESHOLD and (now - last_upload_time >= UPLOAD_INTERVAL):
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
                curr_disease_code = "NO_LEAF"
                curr_display_label = "NO LEAF DETECTED"
                curr_conf = 0.0

            # 4. Display Window Overlays (GUI mode only)
            if not HEADLESS:
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

                cv2.imshow("AgriRover - Continuous Chili Disease Detection", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log("User pressed 'q'. Exiting...")
                    break
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
            cv2.destroyAllWindows()

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
