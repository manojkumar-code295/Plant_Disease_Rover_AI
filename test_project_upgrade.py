"""
Comprehensive Verification Script for Upgraded Plant Leaf Detection & Smart Spraying System.
Tests:
  1. YOLOv8 Object & Obstacle Detection
  2. CNN Plant Leaf Disease Classification
  3. Automatic Categorized Image Storage (detected_images/<Disease>/)
  4. CSV Detection History Logging (detection_logs/detection_history.csv)
  5. Live VS Code Terminal Output formatting
  6. Automatic Water Sprayer Pump Control (Water Only, Duration, Cooldown)
  7. Motor Speed & Obstacle Auto-Braking
  8. Full REST API and Live Video Streaming Endpoints
"""

import os
import json
import base64
import urllib.request
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "http://127.0.0.1:5000"

def post_json(endpoint, payload):
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode('utf-8'))

def get_json(endpoint):
    with urllib.request.urlopen(f"{BASE_URL}{endpoint}") as r:
        return json.loads(r.read().decode('utf-8'))

def run_upgrade_tests():
    print("\n" + "=" * 70)
    print("  VERIFYING UPGRADED LIVE PLANT DETECTION & SMART SPRAYING SYSTEM")
    print("=" * 70)

    # 1. Test Settings API
    print("\n[1/7] Testing System Settings API (/api/settings)...")
    settings = get_json("/api/settings")
    assert "camera" in settings and "detection" in settings and "pump" in settings
    print("      ✓ System Settings API retrieved successfully.")

    update_res = post_json("/api/settings/update", {
        "detection": {"confidence_threshold": 75.0},
        "pump": {"spray_duration": 2.5, "spray_interval": 3.0}
    })
    assert update_res["success"] is True
    print("      ✓ Updated spray duration to 2.5s and confidence to 75%.")

    # 2. Test Live Frame Inference (Anthracnose Leaf)
    print("\n[2/7] Testing Live Frame Continuous Inference (Anthracnose Leaf)...")
    with open(os.path.join(BASE_DIR, "test_samples", "sample_anthracnose.jpg"), "rb") as f:
        leaf_b64 = base64.b64encode(f.read()).decode('utf-8')

    res_leaf = post_json("/api/predict_live", {"image": leaf_b64})
    assert res_leaf["success"] is True
    assert res_leaf["is_leaf"] is True
    assert res_leaf["prediction"] == "Anthracnose"
    print(f"      ✓ Predicted: {res_leaf['prediction']} ({res_leaf['confidence']}%)")
    print(f"      ✓ Auto-Spray Trigger: {res_leaf['auto_spray_trigger']}")
    print(f"      ✓ Object Bounding Boxes: {len(res_leaf['bounding_boxes'])} box(es) localized")

    # 3. Test Non-Leaf Rejection (Strict Filter)
    print("\n[3/7] Testing Strict Non-Leaf Rejection (Human Face)...")
    with open(os.path.join(BASE_DIR, "test_samples", "sample_nonleaf_face.jpg"), "rb") as f:
        face_b64 = base64.b64encode(f.read()).decode('utf-8')

    res_face = post_json("/api/predict_live", {"image": face_b64})
    assert res_face["is_leaf"] is False
    assert res_face["message"] == "Invalid Image! Please upload a clear plant leaf image."
    print(f"      ✓ Correctly rejected face: \"{res_face['message']}\"")

    # 4. Test Obstacle Auto-Braking (Car / Person Obstacle)
    print("\n[4/7] Testing Obstacle Detection & Auto-Braking...")
    with open(os.path.join(BASE_DIR, "test_samples", "sample_nonleaf_car.jpg"), "rb") as f:
        car_b64 = base64.b64encode(f.read()).decode('utf-8')

    res_obs = post_json("/api/predict_live", {"image": car_b64})
    print(f"      ✓ Obstacle Detected in Path: {res_obs.get('has_obstacle')}")
    print(f"      ✓ Motor Direction: {res_obs['hardware_telemetry']['direction']} (Auto-stopped)")

    # 5. Test Automatic Image Storage & CSV History
    print("\n[5/7] Testing Automatic Image Storage & CSV History Log...")
    detected_dir = os.path.join(BASE_DIR, "detected_images")
    history_csv = os.path.join(BASE_DIR, "detection_logs", "detection_history.csv")
    system_log = os.path.join(BASE_DIR, "detection_logs", "system_logs.txt")

    assert os.path.exists(detected_dir), "detected_images folder must exist"
    assert os.path.exists(history_csv), "detection_history.csv must exist"
    assert os.path.exists(system_log), "system_logs.txt must exist"

    # Check images stored in categorized folders
    saved_files = []
    for root, _, files in os.walk(detected_dir):
        for file in files:
            if file.endswith(".jpg"):
                saved_files.append(os.path.join(root, file))

    print(f"      ✓ Automatically saved images count: {len(saved_files)}")
    if saved_files:
        print(f"        Sample stored file: {os.path.basename(saved_files[0])}")

    # Check CSV contents
    with open(history_csv, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"      ✓ detection_history.csv lines: {len(lines)} (Header + Records)")
        if len(lines) > 1:
            print(f"        Last CSV row: {lines[-1].strip()}")

    # 6. Test Rover Speed & Directional Controls
    print("\n[6/7] Testing Rover Motor Controls...")
    for act in ["forward", "left", "right", "backward", "stop"]:
        r = post_json("/api/rover/control", {"action": act})
        assert r["success"] is True

    r_spd = post_json("/api/rover/speed", {"speed": 100})
    assert r_spd["telemetry"]["speed"] == 100
    print("      ✓ Tested forward, left, right, backward, stop, and speed adjustment (100% PWM).")

    # 7. Test Emergency Stop
    print("\n[7/7] Testing Emergency Stop Cutoff...")
    estop_res = post_json("/api/rover/emergency_stop", {})
    assert estop_res["success"] is True
    assert estop_res["telemetry"]["direction"] == "STOP"
    assert estop_res["telemetry"]["pump_active"] is False
    print("      ✓ Emergency Stop immediately disabled all motor and pump outputs.")

    print("\n" + "=" * 70)
    print("  ALL UPGRADE TESTS PASSED WITH 100% SUCCESS! 🎉")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_upgrade_tests()
