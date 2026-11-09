#!/usr/bin/env python3
import os
import time
import threading
import traceback
import requests
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from picamera2 import Picamera2

# ============================================================
# PRECISION AGRICULTURE SMART ROVER - MASTER RASPBERRY PI CODE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_chili_model.keras")
CLASS_FILE = os.path.join(BASE_DIR, "class_names.txt")

RENDER_URL = "https://precision-agriculture-smart-rover.onrender.com"
UPDATE_URL = RENDER_URL + "/api/update-result"
HEARTBEAT_URL = RENDER_URL + "/api/rover/heartbeat"
COMMAND_URL = RENDER_URL + "/api/rover/command"

IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 60.0
UPLOAD_INTERVAL = 1.0
HEARTBEAT_INTERVAL = 5.0
COMMAND_INTERVAL = 0.5
REQUEST_TIMEOUT = 15

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
DISPLAY_CAMERA = True

# ------------------------------------------------------------
# VERIFIED RASPBERRY PI 4 B GPIO PIN MAPPING
# ------------------------------------------------------------
GPIO_ENABLED = True
OBSTACLE_SENSOR_ENABLED = False
OBSTACLE_STOP_DISTANCE_CM = 35.0

# Verified conflict-free GPIO pins
LEFT_IN1 = 17   # Physical Pin 11 (L298N IN1)
LEFT_IN2 = 27   # Physical Pin 13 (L298N IN2)
LEFT_EN  = 12   # Physical Pin 32 (L298N ENA - PWM0)
RIGHT_IN1 = 22  # Physical Pin 15 (L298N IN3)
RIGHT_IN2 = 23  # Physical Pin 16 (L298N IN4)
RIGHT_EN  = 13  # Physical Pin 33 (L298N ENB - PWM1)
PUMP_PIN  = 26  # Physical Pin 37 (Spray Pump Relay IN)
TRIG_PIN  = 24  # Physical Pin 18 (HC-SR04 Trig)
ECHO_PIN  = 25  # Physical Pin 22 (HC-SR04 Echo via Divider)

DISEASE_INFO = {
    "bacterial spot": {
        "chemical": "Copper oxychloride-based bactericide",
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Remove infected leaves and avoid excess moisture.",
    },
    "cercospora leaf spot": {
        "chemical": "Copper-based fungicide",
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Remove infected leaves and maintain good air circulation.",
    },
    "curl virus": {
        "chemical": "Insecticide for whitefly/vector control",
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Remove severely affected leaves and control insect vectors.",
    },
    "healthy leaf": {
        "chemical": "No chemical treatment required",
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Leaf appears healthy. Continue normal monitoring.",
    },
}

running = True
session = requests.Session()
last_command_id = -1
last_pump_state = False
current_direction = "Stopped"
current_speed = 0
current_pump = False
last_upload_time = 0.0
last_heartbeat_time = 0.0

gpio_available = False
lgpio = None
gpio_chip = None


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def normalize_label(label):
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


def load_classes():
    if not os.path.exists(CLASS_FILE):
        raise FileNotFoundError(CLASS_FILE)
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        names = [x.strip() for x in f if x.strip()]
    if not names:
        raise RuntimeError("class_names.txt is empty")
    return names


def load_ai():
    classes = load_classes()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    log("Loading AI model...")
    model = load_model(MODEL_PATH, compile=False)
    outputs = int(model.output_shape[-1])
    if outputs != len(classes):
        raise RuntimeError(
            f"Model has {outputs} outputs but class_names.txt has {len(classes)} classes."
        )
    log(f"AI ready: input={model.input_shape}, output={model.output_shape}")
    log("Classes: " + ", ".join(classes))
    return model, classes


def detect_leaf(frame):
    if frame is None:
        return None, None, 0.0

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([25, 55, 35], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None, 0.0

    frame_area = float(w * h)
    best = None
    best_score = -1.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / frame_area
        if area_ratio < 0.015 or area_ratio > 0.65:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 60 or ch < 40:
            continue

        aspect = cw / float(ch)
        if aspect < 0.15 or aspect > 6.0:
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0:
            continue

        solidity = area / hull_area
        extent = area / float(cw * ch)
        green_pixels = cv2.countNonZero(mask[y:y+ch, x:x+cw])
        green_ratio = green_pixels / float(cw * ch)

        if green_ratio < 0.35 or solidity < 0.55:
            continue
        if extent < 0.20 or extent > 0.95:
            continue

        score = area_ratio * 2.0 + solidity * 0.5 + extent * 0.5 + green_ratio * 2.0
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    if best is None:
        return None, None, 0.0

    x, y, cw, ch = best
    pad = 15
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + cw + pad)
    y2 = min(h, y + ch + pad)

    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2), best_score


def predict(model, classes, leaf):
    image = cv2.resize(
        leaf, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    probabilities = np.asarray(model.predict(image, verbose=0)[0])
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index] * 100.0)
    return classes[index], confidence


def send_result(leaf, disease, confidence):
    global last_upload_time

    now = time.monotonic()
    if now - last_upload_time < UPLOAD_INTERVAL:
        return

    ok, encoded = cv2.imencode(
        ".jpg", leaf, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    if not ok:
        log("Could not encode leaf image")
        return

    info = DISEASE_INFO.get(
        normalize_label(disease),
        {
            "chemical": "Not available",
            "fertilizer": "Balanced NPK fertilizer",
            "treatment": "No treatment information available.",
        },
    )

    files = {"image": ("leaf.jpg", encoded.tobytes(), "image/jpeg")}
    data = {
        "disease": disease,
        "chemical": info["chemical"],
        "fertilizer": info["fertilizer"],
        "treatment": info["treatment"],
        "confidence": f"{confidence:.2f}",
    }

    try:
        r = session.post(UPDATE_URL, files=files, data=data, timeout=REQUEST_TIMEOUT)
        if r.ok:
            last_upload_time = now
            log(f"RESULT SENT -> {disease} | {confidence:.2f}%")
        else:
            log(f"RESULT UPLOAD FAILED -> HTTP {r.status_code}")
    except requests.RequestException as e:
        log(f"RESULT UPLOAD ERROR -> {e}")


def heartbeat():
    global last_heartbeat_time
    now = time.monotonic()
    if now - last_heartbeat_time < HEARTBEAT_INTERVAL:
        return

    try:
        r = session.post(HEARTBEAT_URL, timeout=REQUEST_TIMEOUT)
        if r.ok:
            last_heartbeat_time = now
            log("HEARTBEAT -> OK")
        else:
            log(f"HEARTBEAT -> HTTP {r.status_code}")
    except requests.RequestException as e:
        log(f"HEARTBEAT ERROR -> {e}")


def setup_gpio():
    global gpio_available, lgpio, gpio_chip

    if not GPIO_ENABLED:
        log("GPIO disabled: software-only mode")
        return

    pins = [
        LEFT_IN1, LEFT_IN2, LEFT_EN,
        RIGHT_IN1, RIGHT_IN2, RIGHT_EN,
        PUMP_PIN,
    ]
    if any(p is None for p in pins):
        raise RuntimeError(
            "GPIO_ENABLED=True but real GPIO pins are not configured."
        )

    try:
        import lgpio as _lgpio
        lgpio = _lgpio
        gpio_chip = lgpio.gpiochip_open(0)

        for pin in pins:
            lgpio.gpio_claim_output(gpio_chip, pin, 0)

        gpio_available = True
        log("GPIO initialized")
    except Exception as e:
        raise RuntimeError(f"GPIO initialization failed: {e}")


def gpio_write(pin, value):
    if gpio_available and pin is not None:
        lgpio.gpio_write(gpio_chip, pin, 1 if value else 0)


def stop_motors():
    if not gpio_available:
        return
    gpio_write(LEFT_IN1, 0)
    gpio_write(LEFT_IN2, 0)
    gpio_write(RIGHT_IN1, 0)
    gpio_write(RIGHT_IN2, 0)
    gpio_write(LEFT_EN, 0)
    gpio_write(RIGHT_EN, 0)


def set_pump(state):
    if gpio_available:
        gpio_write(PUMP_PIN, state)


def apply_motor(command, speed):
    global current_direction, current_speed

    command = str(command).lower().strip()
    try:
        speed = max(0, min(100, int(speed)))
    except Exception:
        speed = 0

    if command in ("stop", "emergency_stop"):
        stop_motors()
        current_direction = "Stopped"
        current_speed = 0
        log("MOTOR -> STOP")
        return

    if command == "forward":
        left_forward, right_forward = True, True
    elif command == "backward":
        left_forward, right_forward = False, False
    elif command == "left":
        left_forward, right_forward = False, True
    elif command == "right":
        left_forward, right_forward = True, False
    else:
        stop_motors()
        current_direction = "Stopped"
        current_speed = 0
        log(f"Unknown command {command} -> STOP")
        return

    if gpio_available:
        gpio_write(LEFT_IN1, left_forward)
        gpio_write(LEFT_IN2, not left_forward)
        gpio_write(RIGHT_IN1, right_forward)
        gpio_write(RIGHT_IN2, not right_forward)
        gpio_write(LEFT_EN, speed > 0)
        gpio_write(RIGHT_EN, speed > 0)

    current_direction = command
    current_speed = speed
    log(f"MOTOR -> {command.upper()} | speed={speed}")


def obstacle_blocks_motion():
    # Intentionally disabled until exact HC-SR04 wiring is supplied.
    return False


def apply_render_state(data):
    global last_command_id, last_pump_state, current_pump

    try:
        command_id = int(data.get("command_id", 0))
    except Exception:
        command_id = 0

    command = str(data.get("command", "stop")).lower().strip()

    try:
        speed = int(data.get("speed", 0))
    except Exception:
        speed = 0

    pump = bool(data.get("pump", False))

    if command_id != last_command_id:
        last_command_id = command_id

        log(
            f"NEW COMMAND -> id={command_id}, command={command}, "
            f"speed={speed}, pump={pump}"
        )

        if command == "emergency_stop":
            apply_motor("emergency_stop", 0)
            set_pump(False)
            current_pump = False
            last_pump_state = False
            return

        if command != "stop" and obstacle_blocks_motion():
            apply_motor("stop", 0)
        else:
            apply_motor(command, speed)

    if pump != last_pump_state:
        last_pump_state = pump
        current_pump = pump
        set_pump(pump)
        log("PUMP -> " + ("ON" if pump else "OFF"))


def poll_command():
    try:
        r = session.get(COMMAND_URL, timeout=REQUEST_TIMEOUT)
        if not r.ok:
            log(f"COMMAND API -> HTTP {r.status_code}")
            return

        data = r.json()
        if data.get("success", False):
            apply_render_state(data)

    except requests.RequestException as e:
        log(f"COMMAND ERROR -> {e}")
    except ValueError as e:
        log(f"COMMAND JSON ERROR -> {e}")


def communication_worker():
    while running:
        try:
            heartbeat()
            poll_command()
        except Exception as e:
            log(f"COMMUNICATION ERROR -> {e}")
        time.sleep(COMMAND_INTERVAL)


def draw_display(frame, box, disease, confidence, leaf_score):
    display = frame.copy()

    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if disease:
        text = f"{disease}: {confidence:.1f}%"
        cv2.putText(display, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Leaf score: {leaf_score:.2f}", (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        cv2.putText(display, "NO LEAF DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(display, f"Rover: {current_direction}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, f"Pump: {'ON' if current_pump else 'OFF'}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "Q = quit", (20, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return display


def main():
    global running

    print("=" * 70)
    print("PRECISION AGRICULTURE SMART ROVER")
    print("RASPBERRY PI MASTER PROGRAM")
    print("=" * 70)
    print("Render:", RENDER_URL)
    print("GPIO enabled:", GPIO_ENABLED)
    print("=" * 70)

    model, classes = load_ai()
    setup_gpio()

    camera = None
    thread = threading.Thread(
        target=communication_worker,
        daemon=True
    )
    thread.start()

    try:
        log("Starting CSI camera...")
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                "format": "RGB888",
            }
        )
        camera.configure(config)
        camera.start()
        time.sleep(2)
        log("CSI camera started")
        log("MASTER PROGRAM RUNNING")

        while running:
            frame_rgb = camera.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            leaf, box, leaf_score = detect_leaf(frame)
            disease = None
            confidence = 0.0

            if leaf is not None:
                try:
                    disease, confidence = predict(model, classes, leaf)
                    log(
                        f"AI -> {disease} | {confidence:.2f}% | "
                        f"leaf score={leaf_score:.2f}"
                    )

                    if confidence >= CONFIDENCE_THRESHOLD:
                        send_result(leaf, disease, confidence)

                except Exception as e:
                    log(f"AI ERROR -> {e}")

            if DISPLAY_CAMERA:
                display = draw_display(
                    frame, box, disease, confidence, leaf_score
                )
                cv2.imshow("Precision Agriculture CSI Camera AI", display)

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    running = False
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        log("Keyboard interrupt")
    except Exception as e:
        log(f"MAIN ERROR -> {e}")
        traceback.print_exc()
    finally:
        running = False

        log("SAFE SHUTDOWN")

        try:
            stop_motors()
        except Exception:
            pass

        try:
            set_pump(False)
        except Exception:
            pass

        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if gpio_available:
            try:
                lgpio.gpiochip_close(gpio_chip)
            except Exception:
                pass

        try:
            session.close()
        except Exception:
            pass

        log("MASTER PROGRAM STOPPED")


if __name__ == "__main__":
    main()
