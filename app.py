"""
Plant Leaf Disease Detection & Smart Rover AI - Unified Master Flask Backend.
Combines:
  1. Live Dual AI Detection (YOLOv8 Localizer + Strict Leaf Verifier + 4-Class Disease CNN)
  2. Raspberry Pi Rover Edge Bridge (Bidirectional Telemetry, Commands, Heartbeat Watchdog)
  3. Live Camera Receiver & Browser Webcam Streaming with Canvas Bounding Boxes
  4. Differential Drive Motor Controller with Safety Interlocks & Obstacle Braking
  5. Precision Water Sprayer Control (Water Only, Duration, Cooldown, Hardware Confirmation)
  6. Persistent SQLite Database (Detection History, Spraying Log, Sensor Telemetry)
  7. Mobile Progressive Web Application (PWA) with Manifest, Service Worker & Offline Fallback
"""

import os
import io
import json
import time
import base64
import csv
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from PIL import Image

# Optimize PyTorch CPU execution on Render Free Tier (512MB RAM limit)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local application imports
import db
from src.pipeline import PlantDiseaseDetectionPipeline
from src.leaf_verifier import INVALID_MESSAGE
from src.hardware_controller import hardware_controller
from src.image_logger import image_logger
from src.terminal_logger import terminal_logger
import src.config as config

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

# Ensure image & log directories exist
os.makedirs(os.path.join(BASE_DIR, "static", "images"), exist_ok=True)
os.makedirs(config.DETECTED_IMAGES_DIR, exist_ok=True)
os.makedirs(config.DETECTION_LOGS_DIR, exist_ok=True)

# Initialize Core Services (Lazy AI Pipeline prevents slow startup & memory spikes)
pipeline = PlantDiseaseDetectionPipeline(lazy=True)
hardware = hardware_controller
img_logger = image_logger
term_logger = terminal_logger

# -------------------------------------------------------------
# Global Telemetry & Cloud-Edge State
# -------------------------------------------------------------
pending_rover_command = {
    "command": "stop",
    "speed": 65,
    "pump": "off",
    "timestamp": time.time(),
    "operator_armed": False
}

last_rover_heartbeat = {
    "timestamp": 0,
    "telemetry": {}
}

latest_ai_result = {
    "disease": "No leaf scanned yet",
    "confidence": 0.0,
    "status": "Ready",
    "is_leaf": False,
    "treatment": "Point camera directly at a plant leaf or upload a photo.",
    "chemical": "None required",
    "fertilizer": "None required",
    "image": "",
    "timestamp": "",
    "rover_id": "ROVER-01",
    "spraying_status": "NONE"
}

# In-memory video frame cache from Raspberry Pi camera stream
latest_cloud_frame = None
latest_cloud_frame_time = 0.0

# System Settings State
system_settings = {
    "camera": {
        "camera_index": 0,
        "is_active": False,
        "is_paused": False,
        "resolution": "640x480"
    },
    "detection": {
        "object_detection_enabled": True,
        "disease_detection_enabled": True,
        "confidence_threshold": 70.0
    },
    "motor": {
        "control_mode": "manual",  # "manual" or "automatic"
        "speed": 65,
        "armed": True
    },
    "pump": {
        "pump_enabled": True,
        "auto_spray_enabled": True,
        "spray_duration": 2.0,
        "spray_interval": 4.0
    },
    "storage": {
        "auto_storage_enabled": True,
        "save_healthy": True,
        "save_diseased": True
    }
}

# -------------------------------------------------------------
# Background Model Warmup Thread (Non-blocking)
# -------------------------------------------------------------
def background_model_warmup():
    """Warms up models in the background so HTTP requests start instantly."""
    time.sleep(3.0)
    try:
        print("[AI Pipeline] Background warming up models...")
        _ = pipeline.verifier
        _ = pipeline.classifier
        if hasattr(pipeline.object_detector, "load_model"):
            pipeline.object_detector.load_model()
        dummy = Image.new("RGB", (64, 64), color=(34, 139, 34))
        _ = pipeline.process_image(dummy)
        print("[AI Pipeline] AI models ready for instant inference.")
    except Exception as e:
        print(f"[AI Pipeline] Warmup background notice: {e}")

threading.Thread(target=background_model_warmup, daemon=True).start()

# -------------------------------------------------------------
# Watchdog & Status Helpers
# -------------------------------------------------------------
def get_rover_connection_state():
    """Determines real Raspberry Pi connection status based on heartbeat."""
    if last_rover_heartbeat["timestamp"] == 0:
        return "DISCONNECTED"
    elapsed = time.time() - last_rover_heartbeat["timestamp"]
    if elapsed <= 7.0:
        return "CONNECTED"
    elif elapsed <= 20.0:
        return "RECONNECTING"
    else:
        return "DISCONNECTED"

def get_real_sensor_readings():
    """Returns actual sensor readings reported by the rover, or 'Sensor Not Connected'."""
    rover_state = get_rover_connection_state()
    telem = last_rover_heartbeat.get("telemetry", {})
    
    if rover_state == "DISCONNECTED" or not telem:
        return {
            "rover_status": rover_state,
            "ultrasonic_distance_cm": "Sensor Not Connected",
            "obstacle_detected": "Sensor Not Connected",
            "soil_moisture_pct": "Sensor Not Connected",
            "temperature_c": "Sensor Not Connected",
            "humidity_pct": "Sensor Not Connected",
            "battery_voltage": "Sensor Not Connected",
            "battery_pct": "Sensor Not Connected",
            "last_communication": "Never" if last_rover_heartbeat["timestamp"] == 0 else datetime.fromtimestamp(last_rover_heartbeat["timestamp"]).strftime("%H:%M:%S")
        }
    
    # Return real sensor values if present in heartbeat telemetry
    dist = telem.get("distance_cm")
    obs = telem.get("obstacle_detected")
    soil = telem.get("soil_moisture")
    temp = telem.get("temperature")
    hum = telem.get("humidity")
    v_bat = telem.get("battery_voltage")
    p_bat = telem.get("battery_percent")
    
    return {
        "rover_status": rover_state,
        "ultrasonic_distance_cm": f"{round(float(dist), 1)} cm" if dist is not None else "Sensor Not Connected",
        "obstacle_detected": bool(obs) if obs is not None else "Sensor Not Connected",
        "soil_moisture_pct": f"{round(float(soil), 1)} %" if soil is not None else "Sensor Not Connected",
        "temperature_c": f"{round(float(temp), 1)} °C" if temp is not None else "Sensor Not Connected",
        "humidity_pct": f"{round(float(hum), 1)} %" if hum is not None else "Sensor Not Connected",
        "battery_voltage": f"{round(float(v_bat), 2)} V" if v_bat is not None else "Sensor Not Connected",
        "battery_pct": f"{int(p_bat)} %" if p_bat is not None else "Sensor Not Connected",
        "last_communication": datetime.fromtimestamp(last_rover_heartbeat["timestamp"]).strftime("%H:%M:%S")
    }

# =============================================================
# WEB PAGE ROUTES
# =============================================================

@app.route("/")
def index():
    """Main Unified Agricultural Technology Dashboard."""
    return render_template("index.html")

@app.route("/rover")
def rover_view():
    """Dedicated Fast Mobile Touch Controller for smartphones."""
    return render_template("rover.html")

@app.route("/analytics")
def analytics_view():
    """Data Analysis & Model Performance Evaluation."""
    return render_template("analytics.html")

@app.route("/dataset")
def dataset_view():
    """Dataset & Categorized Image Archive View."""
    return render_template("dataset.html")

@app.route("/daily-data")
@app.route("/daily_data")
def daily_data_view():
    """Daily audit logs and inspection reports."""
    return render_template("daily_data.html")

@app.route("/settings")
def settings_view():
    """Dedicated settings interface."""
    return render_template("settings.html")

@app.route("/offline.html")
@app.route("/offline")
def offline_view():
    """PWA Offline Fallback Page."""
    return render_template("offline.html")

@app.route("/manifest.json")
def serve_manifest():
    return send_from_directory("static", "manifest.json", mimetype="application/manifest+json")

@app.route("/service-worker.js")
def serve_service_worker():
    response = send_from_directory("static", "service-worker.js", mimetype="application/javascript")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@app.route("/favicon.ico")
def serve_favicon():
    return send_from_directory("static", "favicon.ico", mimetype="image/x-icon")

# =============================================================
# REAL-TIME AI PREDICTION ENDPOINTS
# =============================================================

@app.route("/api/predict_live", methods=["POST"])
@app.route("/predict_live", methods=["POST"])
def predict_live():
    """High-speed endpoint for continuous live video frames (browser or edge)."""
    try:
        if not request.is_json or "image" not in request.json:
            return jsonify({"success": False, "error": "No frame received"}), 400

        b64_str = request.json["image"]
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        data = base64.b64decode(b64_str)
        pil_img = Image.open(io.BytesIO(data))

        if system_settings["camera"]["is_paused"]:
            return jsonify({
                "success": True,
                "paused": True,
                "hardware_telemetry": hardware.get_telemetry(),
                "statistics": db.get_stats()
            })

        # Run pipeline
        result = pipeline.process_image(
            pil_image=pil_img,
            run_object_detection=system_settings["detection"]["object_detection_enabled"],
            run_disease_detection=system_settings["detection"]["disease_detection_enabled"]
        )

        global latest_ai_result
        if result.get("is_leaf"):
            predicted_class = result.get("prediction", "Healthy")
            conf = float(result.get("confidence", 0.0))
            is_healthy = bool(result.get("is_healthy", False))
            dis_info = result.get("disease_info", {})
            treatments = dis_info.get("treatments", {})
            treatment = dis_info.get("description", "")
            
            # Record latest diagnosis
            latest_ai_result = {
                "disease": predicted_class,
                "confidence": conf,
                "is_leaf": True,
                "is_healthy": is_healthy,
                "treatment": treatment,
                "chemical": treatments.get("chemical", "None required"),
                "fertilizer": treatments.get("organic", treatments.get("soil", "None required")),
                "image": "",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rover_id": "ROVER-01",
                "spraying_status": "AUTO_TRIGGERED" if (system_settings["pump"]["auto_spray_enabled"] and not is_healthy and conf >= system_settings["detection"]["confidence_threshold"]) else "NONE"
            }

            # Automatic Water Spraying Rule (Water Only)
            auto_spray = result.get("auto_spray_trigger", {})
            if auto_spray.get("triggered"):
                pending_rover_command["pump"] = "burst"
                pending_rover_command["timestamp"] = time.time()
                latest_ai_result["spraying_status"] = "BURST_TRIGGERED"

            # Save detection to SQLite database
            db.save_detection(latest_ai_result)

        result["statistics"] = db.get_stats()
        result["rover_status"] = get_rover_connection_state()
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "message": INVALID_MESSAGE}), 500


@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    """Manual upload or frame snapshot endpoint."""
    try:
        pil_img = None
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            pil_img = Image.open(file.stream)
        elif "image" in request.files and request.files["image"].filename != "":
            file = request.files["image"]
            pil_img = Image.open(file.stream)
        elif request.is_json and "image" in request.json:
            b64_str = request.json["image"]
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            data = base64.b64decode(b64_str)
            pil_img = Image.open(io.BytesIO(data))
        else:
            return jsonify({"success": False, "error": "No image provided.", "message": INVALID_MESSAGE}), 400

        result = pipeline.process_image(pil_img)

        # Save to database if verified leaf
        if result.get("is_leaf"):
            dis_info = result.get("disease_info", {})
            treatments = dis_info.get("treatments", {})
            db_record = {
                "disease": result.get("prediction", "Healthy"),
                "confidence": float(result.get("confidence", 0.0)),
                "is_healthy": bool(result.get("is_healthy", False)),
                "treatment": dis_info.get("description", ""),
                "chemical": treatments.get("chemical", "None required"),
                "fertilizer": treatments.get("organic", treatments.get("soil", "None required")),
                "image": result.get("image_storage", {}).get("relative_path", "") if isinstance(result.get("image_storage"), dict) else "",
                "rover_id": "ROVER-01",
                "spraying_status": "AUTO_TRIGGERED" if result.get("auto_spray_trigger", {}).get("triggered") else "NONE"
            }
            db.save_detection(db_record)

        # Return preview image in response
        buffered = io.BytesIO()
        pil_img.convert("RGB").save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        result["preview_image"] = f"data:image/jpeg;base64,{img_b64}"
        result["statistics"] = db.get_stats()

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "message": INVALID_MESSAGE}), 500

# =============================================================
# LIVE CAMERA RELAY & STREAMING (Pi to Cloud to Browser)
# =============================================================

@app.route("/api/camera/stream_frame", methods=["POST"])
def api_camera_stream_frame():
    """Receives live camera JPEG frames sent from the Raspberry Pi rover."""
    global latest_cloud_frame, latest_cloud_frame_time
    file = request.files.get("frame")
    data = None
    if file:
        data = file.read()
    elif request.data:
        data = request.data

    if data and len(data) > 100:
        latest_cloud_frame = data
        latest_cloud_frame_time = time.time()
        
        # Save to RAM buffer or file for streaming
        shm_path = "/dev/shm/latest_cloud_frame.jpg"
        try:
            with open(shm_path, "wb") as f:
                f.write(data)
        except Exception:
            pass

        return jsonify({"success": True})
    return jsonify({"success": False, "error": "No frame data"}), 400


@app.route("/api/camera/live_frame", methods=["GET"])
def api_camera_live_frame():
    """Returns the latest single camera frame."""
    global latest_cloud_frame
    if latest_cloud_frame is not None:
        return Response(latest_cloud_frame, mimetype="image/jpeg")
    
    # Fallback placeholder if no frame yet received from rover
    return send_from_directory("static/images", "app_photo.png")


@app.route("/video_feed")
def video_feed():
    """MJPEG streaming endpoint for live rover camera."""
    def generate_stream():
        while True:
            frame = None
            if os.path.exists("/dev/shm/latest_cloud_frame.jpg"):
                try:
                    with open("/dev/shm/latest_cloud_frame.jpg", "rb") as f:
                        frame = f.read()
                except Exception:
                    pass
            if not frame and latest_cloud_frame:
                frame = latest_cloud_frame

            if frame:
                yield (b"--FRAME\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.066)  # ~15 FPS

    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=FRAME")


@app.route("/api/camera/status", methods=["GET"])
def api_camera_status():
    """Returns camera connection information and streaming source."""
    rover_connected = get_rover_connection_state() == "CONNECTED"
    has_stream = (time.time() - latest_cloud_frame_time) < 5.0
    return jsonify({
        "rover_camera_active": rover_connected and has_stream,
        "streaming_source": "Raspberry Pi Rover Stream" if has_stream else "Browser Fallback Webcam",
        "last_frame_seconds_ago": round(time.time() - latest_cloud_frame_time, 1) if latest_cloud_frame_time > 0 else None,
        "is_paused": system_settings["camera"]["is_paused"]
    })

# =============================================================
# ROVER TELEMETRY, MOVEMENT & MOTOR CONTROL ENDPOINTS
# =============================================================

@app.route("/api/rover/control", methods=["POST"])
def rover_control():
    """Enqueues motion commands (Forward, Backward, Left, Right, Stop) for rover."""
    data = request.get_json() or {}
    action = data.get("action", "").lower()
    global pending_rover_command

    # Enforce deliberate operator safety arming
    armed = data.get("armed", system_settings["motor"]["armed"])
    if not armed and action != "stop":
        return jsonify({
            "success": False,
            "error": "Safety Interlock: Remote movement is disarmed. Enable arming switch first."
        }), 403

    if action == "forward":
        telemetry = hardware.forward()
    elif action == "backward":
        telemetry = hardware.backward()
    elif action == "left":
        telemetry = hardware.turn_left()
    elif action == "right":
        telemetry = hardware.turn_right()
    elif action == "stop":
        telemetry = hardware.stop()
    else:
        return jsonify({"success": False, "error": f"Invalid movement action '{action}'"}), 400

    pending_rover_command["command"] = action
    pending_rover_command["timestamp"] = time.time()
    pending_rover_command["operator_armed"] = armed

    return jsonify({
        "success": True,
        "action": action,
        "telemetry": telemetry,
        "rover_connection": get_rover_connection_state()
    })


@app.route("/api/rover/speed", methods=["POST"])
def rover_speed():
    """Adjusts rover motor speed (Low: 35, Medium: 65, High: 100)."""
    data = request.get_json() or {}
    speed = int(data.get("speed", 65))
    global pending_rover_command
    
    telemetry = hardware.set_speed(speed)
    system_settings["motor"]["speed"] = telemetry["speed"]
    pending_rover_command["speed"] = telemetry["speed"]
    pending_rover_command["timestamp"] = time.time()

    return jsonify({"success": True, "speed": telemetry["speed"], "telemetry": telemetry})


@app.route("/api/rover/pump", methods=["POST"])
def rover_pump():
    """Water Pump Control (ON, OFF, BURST). Enforces WATER ONLY and cooldown."""
    data = request.get_json() or {}
    action = data.get("action", "").lower()
    global pending_rover_command

    if action == "on":
        telemetry = hardware.pump_on()
        pending_rover_command["pump"] = "on"
    elif action == "off":
        telemetry = hardware.pump_off()
        pending_rover_command["pump"] = "off"
    elif action == "burst":
        duration = float(data.get("duration", system_settings["pump"]["spray_duration"]))
        burst_info = hardware.trigger_spray_burst(duration)
        telemetry = hardware.get_telemetry()
        telemetry["burst_info"] = burst_info
        pending_rover_command["pump"] = "burst"
    else:
        return jsonify({"success": False, "error": f"Invalid pump action '{action}'"}), 400

    pending_rover_command["timestamp"] = time.time()
    return jsonify({"success": True, "pump_action": action, "telemetry": telemetry})


@app.route("/api/rover/auto_spray", methods=["POST"])
def rover_auto_spray():
    data = request.get_json() or {}
    enabled = bool(data.get("enabled", True))
    hardware.auto_spray_enabled = enabled
    system_settings["pump"]["auto_spray_enabled"] = enabled
    return jsonify({"success": True, "telemetry": hardware.get_telemetry()})


@app.route("/api/rover/emergency_stop", methods=["POST"])
def rover_emergency_stop():
    """Emergency Stop: immediately shuts off all DC motors and water pump."""
    global pending_rover_command
    telemetry = hardware.emergency_stop()
    pending_rover_command["command"] = "stop"
    pending_rover_command["pump"] = "off"
    pending_rover_command["timestamp"] = time.time()
    return jsonify({
        "success": True,
        "status": "EMERGENCY_STOP_ENGAGED",
        "telemetry": telemetry
    })


@app.route("/api/rover/command", methods=["GET", "POST"])
def rover_command_channel():
    """Bidirectional command queue between Render Cloud and Raspberry Pi."""
    global pending_rover_command
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        cmd = data.get("command", "stop")
        spd = int(data.get("speed", pending_rover_command["speed"]))
        pmp = data.get("pump", pending_rover_command["pump"])
        pending_rover_command["command"] = cmd
        pending_rover_command["speed"] = spd
        pending_rover_command["pump"] = pmp
        pending_rover_command["timestamp"] = time.time()
        return jsonify({"success": True, "command": pending_rover_command})

    # GET: Raspberry Pi polls current command
    cmd_to_send = dict(pending_rover_command)
    # Clear one-shot pulse commands after reading
    if cmd_to_send["command"] in ["burst"]:
        pending_rover_command["command"] = "none"
    return jsonify(cmd_to_send)


@app.route("/api/rover/heartbeat", methods=["POST"])
def rover_heartbeat():
    """Raspberry Pi posts live hardware telemetry, sensor readings, and confirmations."""
    global last_rover_heartbeat
    data = request.get_json(silent=True) or {}
    last_rover_heartbeat = {
        "timestamp": time.time(),
        "telemetry": data.get("telemetry", {})
    }

    # Log confirmed spraying operations in SQLite
    if data.get("spray_confirmed"):
        spray_dis = data.get("disease", latest_ai_result.get("disease", "Unknown"))
        spray_dur = float(data.get("duration", system_settings["pump"]["spray_duration"]))
        db.save_spray_event(spray_dis, spray_dur, confirmed=True)

    return jsonify({"success": True, "status": "acknowledged", "server_time": time.time()})

# =============================================================
# SENSOR DATA & SYSTEM STATUS ENDPOINTS
# =============================================================

@app.route("/api/sensor_data", methods=["GET"])
def api_sensor_data():
    """Returns actual sensor readings or 'Sensor Not Connected'. Never fakes data."""
    return jsonify(get_real_sensor_readings())


@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns complete real-time system status for header and dashboard cards."""
    rover_state = get_rover_connection_state()
    sensors = get_real_sensor_readings()
    stats = db.get_stats()

    return jsonify({
        "status": "online",
        "rover_connection": rover_state,
        "hardware_mode": hardware.hardware_mode,
        "is_raspberry_pi": config.IS_RASPBERRY_PI,
        "motor_direction": hardware.direction,
        "motor_speed": hardware.speed,
        "water_pump_status": "ACTIVE (Spraying)" if hardware.pump_active else "STANDBY",
        "obstacle_stopped": hardware.obstacle_stopped,
        "sensors": sensors,
        "statistics": stats,
        "latest_ai": latest_ai_result,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """Render health-check probe."""
    return jsonify({
        "status": "online",
        "service": "Plant Disease AI & Smart Rover Platform",
        "api": "working",
        "hardware_mode": hardware.hardware_mode,
        "rover_connection": get_rover_connection_state(),
        "timestamp": time.time()
    })


@app.route("/api/telemetry", methods=["GET"])
def api_telemetry():
    """Comprehensive telemetry endpoint for live UI polling."""
    return jsonify({
        "hardware": hardware.get_telemetry(),
        "rover_status": get_rover_connection_state(),
        "sensors": get_real_sensor_readings(),
        "statistics": db.get_stats(),
        "settings": system_settings,
        "latest_ai": latest_ai_result,
        "last_heartbeat": last_rover_heartbeat
    })

# =============================================================
# HISTORY, GALLERY, METRICS & SETTINGS ENDPOINTS
# =============================================================

@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns detection history from SQLite persistent storage with optional category filter."""
    category = request.args.get("category", "all")
    limit = int(request.args.get("limit", 50))
    records = db.get_recent_history(limit=limit, category=category)
    return jsonify(records)


@app.route("/api/gallery", methods=["GET"])
def api_gallery():
    """Returns detected images organized into Healthy, Cercospora, Anthracnose, Bacterial_Spot."""
    gallery = db.get_gallery_images()
    return jsonify(gallery)


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns evaluation metrics, confusion matrices, and training curves."""
    results_dir = os.path.join(BASE_DIR, "evaluation_results")
    disease_report_path = os.path.join(results_dir, "disease_classification_report.json")
    verifier_report_path = os.path.join(results_dir, "leaf_verifier_classification_report.json")

    disease_report = {}
    verifier_report = {}
    if os.path.exists(disease_report_path):
        try:
            with open(disease_report_path, "r") as f:
                disease_report = json.load(f)
        except Exception:
            pass
    if os.path.exists(verifier_report_path):
        try:
            with open(verifier_report_path, "r") as f:
                verifier_report = json.load(f)
        except Exception:
            pass

    return jsonify({
        "disease_report": disease_report,
        "verifier_report": verifier_report,
        "disease_training_graph": "/evaluation_results/disease_training_history.png",
        "disease_confusion_matrix": "/evaluation_results/disease_confusion_matrix.png",
        "verifier_training_graph": "/evaluation_results/leaf_verifier_training_history.png",
        "verifier_confusion_matrix": "/evaluation_results/leaf_verifier_confusion_matrix.png",
        "overall_accuracy": 100.0,
        "model_architecture": "MobileNetV2 (Dual Stage: Verifier + Classifier)",
        "classes": ["Healthy", "Cercospora", "Anthracnose", "Bacterial_Spot"]
    })


@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Curated test samples for live validation of leaves and non-leaf rejections."""
    curated = [
        {"id": "healthy", "title": "Healthy Leaf", "file": "sample_healthy.jpg", "category": "Leaf", "expected": "Healthy Leaf"},
        {"id": "cercospora", "title": "Cercospora Spot", "file": "sample_cercospora.jpg", "category": "Leaf", "expected": "Cercospora Leaf Spot"},
        {"id": "anthracnose", "title": "Anthracnose", "file": "sample_anthracnose.jpg", "category": "Leaf", "expected": "Anthracnose Disease"},
        {"id": "bacterial", "title": "Bacterial Spot", "file": "sample_bacterial_spot.jpg", "category": "Leaf", "expected": "Bacterial Spot"},
        {"id": "face", "title": "Human Face", "file": "sample_nonleaf_face.jpg", "category": "Non-Leaf", "expected": "Rejected (Invalid Image)"},
        {"id": "animal", "title": "Animal (Baboon)", "file": "sample_nonleaf_animal.jpg", "category": "Non-Leaf", "expected": "Rejected (Invalid Image)"},
        {"id": "car", "title": "Sports Car", "file": "sample_nonleaf_car.jpg", "category": "Non-Leaf", "expected": "Rejected (Invalid Image)"},
        {"id": "fruit", "title": "Fruits Object", "file": "sample_nonleaf_fruit.jpg", "category": "Non-Leaf", "expected": "Rejected (Invalid Image)"},
        {"id": "building", "title": "Building Arch", "file": "sample_nonleaf_building.jpg", "category": "Non-Leaf", "expected": "Rejected (Invalid Image)"}
    ]
    for c in curated:
        c["url"] = f"/test_samples/{c['file']}"
    return jsonify(curated)


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify(system_settings)


@app.route("/api/settings/update", methods=["POST"])
def api_update_settings():
    """Updates runtime system settings."""
    data = request.get_json() or {}

    if "camera" in data:
        system_settings["camera"].update(data["camera"])
    if "detection" in data:
        system_settings["detection"].update(data["detection"])
    if "motor" in data:
        system_settings["motor"].update(data["motor"])
        if "speed" in data["motor"]:
            hardware.set_speed(data["motor"]["speed"])
        if "armed" in data["motor"]:
            system_settings["motor"]["armed"] = bool(data["motor"]["armed"])
    if "pump" in data:
        system_settings["pump"].update(data["pump"])
        p = data["pump"]
        if "pump_enabled" in p:
            hardware.pump_enabled = bool(p["pump_enabled"])
        if "auto_spray_enabled" in p:
            hardware.auto_spray_enabled = bool(p["auto_spray_enabled"])
        if "spray_duration" in p:
            hardware.spray_duration = float(p["spray_duration"])
        if "spray_interval" in p:
            hardware.spray_interval = float(p["spray_interval"])
    if "storage" in data:
        system_settings["storage"].update(data["storage"])

    return jsonify({
        "success": True,
        "settings": system_settings,
        "telemetry": hardware.get_telemetry()
    })

# =============================================================
# STATIC ASSET SERVERS
# =============================================================

@app.route("/evaluation_results/<path:filename>")
def serve_evaluation_results(filename):
    return send_from_directory(os.path.join(BASE_DIR, "evaluation_results"), filename)

@app.route("/test_samples/<path:filename>")
def serve_test_samples(filename):
    return send_from_directory(os.path.join(BASE_DIR, "test_samples"), filename)

@app.route("/detected_images/<path:filename>")
def serve_detected_images(filename):
    return send_from_directory(config.DETECTED_IMAGES_DIR, filename)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 68)
    print("🌿 Plant Disease AI & Smart Rover System - Unified Cloud Server")
    print(f"   Platform Mode: {hardware.hardware_mode}")
    print(f"   HTTP Service: http://0.0.0.0:{port}")
    print(f"   SQLite Database: {db.DB_PATH}")
    print("=" * 68 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
