"""
AgriVision AI - Plant Leaf Detection & Automatic Smart Spraying System.
Flask Application Backend.
Features:
  - Continuous Live Webcam Detection & Real-time Canvas Bounding Box Streaming
  - Dual Object Detection (YOLOv8) & Deep Disease Classification CNN
  - Live VS Code Terminal Output
  - Automatic Precision Water Spraying (Water Only, Duration & Cooldown Control)
  - Automatic Categorized Image Archival (Healthy, Cercospora, Anthracnose, Bacterial_Spot)
  - Differential Drive Motor Navigation & Obstacle Auto-Braking
  - Hardware & Simulation Dual-Mode Execution
"""

import os
import io
import json
import time
import base64
import csv
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from src.pipeline import PlantDiseaseDetectionPipeline
from src.leaf_verifier import INVALID_MESSAGE
from src.hardware_controller import hardware_controller
from src.image_logger import image_logger
from src.terminal_logger import terminal_logger
import src.config as config

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Initialize AI Pipeline
pipeline = PlantDiseaseDetectionPipeline()
hardware = hardware_controller
img_logger = image_logger
term_logger = terminal_logger

# Global state for cloud-edge synchronization
pending_rover_command = {"command": "none", "speed": 65, "pump": "off", "timestamp": 0}
last_rover_heartbeat = {"timestamp": 0, "telemetry": {}}
latest_ai_result = {
    "disease": "No leaf scanned yet",
    "confidence": 0.0,
    "status": "Ready",
    "treatment": "Waiting for live camera frame...",
    "timestamp": ""
}

# Runtime Settings State
system_settings = {
    "camera": {
        "camera_index": 0,
        "is_active": False,
        "is_paused": False
    },
    "detection": {
        "object_detection_enabled": True,
        "disease_detection_enabled": True,
        "confidence_threshold": 70.0
    },
    "motor": {
        "control_mode": "manual",  # "manual" or "automatic"
        "speed": 65
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

@app.route("/")
def index():
    return render_template("index.html")

# ==========================================
# REAL-TIME AI PREDICTION ENDPOINTS
# ==========================================

@app.route("/api/predict_live", methods=["POST"])
def predict_live():
    """High-speed endpoint for continuous live webcam video frames."""
    try:
        if not request.is_json or "image" not in request.json:
            return jsonify({"success": False, "error": "No frame received"}), 400

        b64_str = request.json["image"]
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        data = base64.b64decode(b64_str)
        pil_img = Image.open(io.BytesIO(data))

        # Check if paused
        if system_settings["camera"]["is_paused"]:
            return jsonify({
                "success": True,
                "paused": True,
                "hardware_telemetry": hardware.get_telemetry(),
                "statistics": img_logger.get_stats()
            })

        # Process frame through upgraded pipeline
        result = pipeline.process_image(
            pil_image=pil_img,
            run_object_detection=system_settings["detection"]["object_detection_enabled"],
            run_disease_detection=system_settings["detection"]["disease_detection_enabled"]
        )

        global latest_ai_result
        leaf_res = result.get("leaf_disease")
        if leaf_res and leaf_res.get("is_leaf"):
            latest_ai_result = {
                "disease": leaf_res.get("predicted_class", "Healthy"),
                "confidence": leaf_res.get("confidence", 0.0),
                "is_leaf": True,
                "treatment": leaf_res.get("treatment_notes", ""),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "message": INVALID_MESSAGE}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    """Manual upload or frame snapshot endpoint."""
    try:
        pil_img = None
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
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

        # Return preview image in response
        buffered = io.BytesIO()
        pil_img.convert("RGB").save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        result["preview_image"] = f"data:image/jpeg;base64,{img_b64}"

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "message": INVALID_MESSAGE}), 500

# ==========================================
# HARDWARE MOTOR & PUMP ENDPOINTS
# ==========================================

@app.route("/api/rover/control", methods=["POST"])
def rover_control():
    data = request.get_json() or {}
    action = data.get("action", "").lower()
    global pending_rover_command

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
        return jsonify({"success": False, "error": f"Invalid action '{action}'"}), 400

    pending_rover_command["command"] = action
    pending_rover_command["timestamp"] = time.time()
    return jsonify({"success": True, "telemetry": telemetry})

@app.route("/api/rover/speed", methods=["POST"])
def rover_speed():
    data = request.get_json() or {}
    speed = data.get("speed", 65)
    global pending_rover_command
    telemetry = hardware.set_speed(speed)
    system_settings["motor"]["speed"] = telemetry["speed"]
    pending_rover_command["speed"] = telemetry["speed"]
    pending_rover_command["timestamp"] = time.time()
    return jsonify({"success": True, "telemetry": telemetry})

@app.route("/api/rover/pump", methods=["POST"])
def rover_pump():
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
    return jsonify({"success": True, "telemetry": telemetry})

@app.route("/api/rover/emergency_stop", methods=["POST"])
def rover_emergency_stop():
    global pending_rover_command
    telemetry = hardware.emergency_stop()
    pending_rover_command["command"] = "stop"
    pending_rover_command["pump"] = "off"
    pending_rover_command["timestamp"] = time.time()
    return jsonify({"success": True, "telemetry": telemetry})

@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    return jsonify({
        "hardware": hardware.get_telemetry(),
        "statistics": img_logger.get_stats(),
        "settings": system_settings,
        "latest_ai": latest_ai_result,
        "last_heartbeat": last_rover_heartbeat
    })

# ==========================================
# CLOUD & RASPBERRY PI BRIDGE ENDPOINTS
# ==========================================

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "Precision Agriculture Smart Rover & AgriVision AI",
        "api": "working",
        "hardware_mode": hardware.hardware_mode
    })

@app.route("/api/result", methods=["GET"])
def get_result():
    return jsonify(latest_ai_result)

@app.route("/api/rover/command", methods=["GET"])
def get_rover_command():
    global pending_rover_command
    # Return current pending command
    cmd_to_send = dict(pending_rover_command)
    # Clear one-shot command after reading if older than 5 seconds
    if cmd_to_send["command"] in ["burst", "stop"]:
        pending_rover_command["command"] = "none"
    return jsonify(cmd_to_send)

@app.route("/api/rover/heartbeat", methods=["POST"])
def rover_heartbeat():
    global last_rover_heartbeat
    data = request.get_json(silent=True) or {}
    last_rover_heartbeat = {
        "timestamp": time.time(),
        "telemetry": data.get("telemetry", {})
    }
    return jsonify({"success": True, "status": "acknowledged"})

@app.route("/api/rover/gpio-config", methods=["GET"])
def rover_gpio_config():
    return jsonify({
        "left_in1": config.MOTOR_LEFT_IN1,
        "left_in2": config.MOTOR_LEFT_IN2,
        "left_en": config.MOTOR_LEFT_ENA,
        "right_in1": config.MOTOR_RIGHT_IN3,
        "right_in2": config.MOTOR_RIGHT_IN4,
        "right_en": config.MOTOR_RIGHT_ENB,
        "pump_pin": config.PUMP_RELAY_PIN,
        "hardware_mode": hardware.hardware_mode,
        "is_raspberry_pi": config.IS_RASPBERRY_PI
    })

@app.route("/api/update-result", methods=["POST"])
def update_result():
    global latest_ai_result
    disease = request.form.get("disease", "Unknown")
    confidence = float(request.form.get("confidence", 0.0))
    treatment = request.form.get("treatment", "")
    is_leaf = request.form.get("is_leaf", "true").lower() == "true"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    latest_ai_result = {
        "disease": disease,
        "confidence": confidence,
        "is_leaf": is_leaf,
        "treatment": treatment,
        "timestamp": timestamp
    }
    
    # Save image if provided
    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            try:
                pil_img = Image.open(file.stream)
                img_logger.save_detection(pil_img, {
                    "is_leaf": is_leaf,
                    "predicted_class": disease,
                    "confidence": confidence,
                    "treatment_notes": treatment
                })
            except Exception as e:
                print(f"[Storage] Error saving uploaded result image: {e}")

    return jsonify({"success": True, "result": latest_ai_result})

# ==========================================
# SYSTEM SETTINGS ENDPOINTS
# ==========================================

@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify(system_settings)

@app.route("/api/settings/update", methods=["POST"])
def update_settings():
    data = request.get_json() or {}

    # Camera Settings
    if "camera" in data:
        system_settings["camera"].update(data["camera"])
        if "is_active" in data["camera"]:
            term_logger.log_camera_status(data["camera"]["is_active"])

    # Detection Settings
    if "detection" in data:
        system_settings["detection"].update(data["detection"])

    # Motor Settings
    if "motor" in data:
        system_settings["motor"].update(data["motor"])
        if "control_mode" in data["motor"]:
            hardware.set_control_mode(data["motor"]["control_mode"])
        if "speed" in data["motor"]:
            hardware.set_speed(data["motor"]["speed"])

    # Pump Settings
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

    # Storage Settings
    if "storage" in data:
        system_settings["storage"].update(data["storage"])
        s = data["storage"]
        if "auto_storage_enabled" in s:
            img_logger.storage_enabled = bool(s["auto_storage_enabled"])
        if "save_healthy" in s:
            img_logger.save_healthy = bool(s["save_healthy"])
        if "save_diseased" in s:
            img_logger.save_diseased = bool(s["save_diseased"])

    return jsonify({"success": True, "settings": system_settings, "telemetry": hardware.get_telemetry()})

@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns the latest detection history from CSV."""
    records = []
    if os.path.exists(config.CSV_LOG_PATH):
        try:
            with open(config.CSV_LOG_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                records = rows[-25:]  # latest 25 detections
                records.reverse()
        except Exception:
            pass
    return jsonify(records)

# ==========================================
# STATIC ASSETS & METRICS ENDPOINTS
# ==========================================

@app.route("/evaluation_results/<path:filename>")
def serve_evaluation_results(filename):
    return send_from_directory(os.path.join(BASE_DIR, "evaluation_results"), filename)

@app.route("/test_samples/<path:filename>")
def serve_test_samples(filename):
    return send_from_directory(os.path.join(BASE_DIR, "test_samples"), filename)

@app.route("/detected_images/<path:filename>")
def serve_detected_images(filename):
    return send_from_directory(config.DETECTED_IMAGES_DIR, filename)

@app.route("/api/samples", methods=["GET"])
def get_samples():
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

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    results_dir = os.path.join(BASE_DIR, "evaluation_results")
    disease_report_path = os.path.join(results_dir, "disease_classification_report.json")
    verifier_report_path = os.path.join(results_dir, "leaf_verifier_classification_report.json")

    disease_report = {}
    verifier_report = {}
    if os.path.exists(disease_report_path):
        with open(disease_report_path, "r") as f:
            disease_report = json.load(f)
    if os.path.exists(verifier_report_path):
        with open(verifier_report_path, "r") as f:
            verifier_report = json.load(f)

    return jsonify({
        "disease_report": disease_report,
        "verifier_report": verifier_report,
        "disease_training_graph": "/evaluation_results/disease_training_history.png",
        "disease_confusion_matrix": "/evaluation_results/disease_confusion_matrix.png",
        "verifier_training_graph": "/evaluation_results/leaf_verifier_training_history.png",
        "verifier_confusion_matrix": "/evaluation_results/leaf_verifier_confusion_matrix.png"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n==================================================================")
    print("🌿 AgriVision AI: Plant Leaf Detection & Smart Rover System")
    print(f"   Hardware Mode: {hardware.hardware_mode}")
    print("   VS Code Terminal Active - Real-Time Logs Will Appear Below")
    print(f"   Open Web Dashboard at: http://127.0.0.1:{port}")
    print("==================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
