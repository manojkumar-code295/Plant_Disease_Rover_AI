"""
Automated Test Script for Smart Rover Controls & Continuous AI Detection.
Verifies:
  - Motor direction commands (Forward, Backward, Left, Right, Stop)
  - Speed throttling (PWM duty cycles)
  - Sprayer Pump relay control (Manual ON/OFF & 2s Precision Burst)
  - Smart Auto-Spray triggering upon AI disease diagnosis
  - Live continuous stream frame processing
"""

import urllib.request
import json
import base64
import time

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

def run_tests():
    print("\n" + "=" * 65)
    print("  TESTING SMART ROVER & CONTINUOUS AI DETECTION ENDPOINTS")
    print("=" * 65)

    # 1. Rover Motor Controls
    print("\n1. Testing Rover Motor Directions:")
    for action in ["forward", "backward", "left", "right", "stop"]:
        res = post_json("/api/rover/control", {"action": action})
        dir_state = res["telemetry"]["direction"]
        print(f"   Command '{action.upper()}': telemetry direction = '{dir_state}' -> PASS")
        assert dir_state.lower() == action.lower()

    # 2. Rover Speed Controls
    print("\n2. Testing Rover Multi-Speed Regulation:")
    for spd in [35, 65, 100]:
        res = post_json("/api/rover/speed", {"speed": spd})
        cur_spd = res["telemetry"]["speed"]
        print(f"   Set Speed {spd}%: telemetry speed = {cur_spd}% -> PASS")
        assert cur_spd == spd

    # 3. Sprayer Pump Controls
    print("\n3. Testing Sprayer Pump Controls:")
    res_on = post_json("/api/rover/pump", {"action": "on"})
    print(f"   Pump ON: pump_active = {res_on['telemetry']['pump_active']} -> PASS")
    assert res_on["telemetry"]["pump_active"] is True

    res_off = post_json("/api/rover/pump", {"action": "off"})
    print(f"   Pump OFF: pump_active = {res_off['telemetry']['pump_active']} -> PASS")
    assert res_off["telemetry"]["pump_active"] is False

    res_burst = post_json("/api/rover/pump", {"action": "burst", "duration": 0.5})
    print(f"   Pump Spray Burst (0.5s): response = {res_burst['success']} -> PASS")
    assert res_burst["success"] is True

    # 4. Auto-Spray Mode
    print("\n4. Testing Smart Auto-Spray Mode Toggle:")
    res_auto = post_json("/api/rover/auto_spray", {"enabled": True})
    print(f"   Auto-Spray ENABLED: telemetry auto_spray_enabled = {res_auto['telemetry']['auto_spray_enabled']} -> PASS")
    assert res_auto["telemetry"]["auto_spray_enabled"] is True

    # 5. Continuous Live Frame Detection (Leaf with Disease)
    print("\n5. Testing Live Frame Continuous Detection (Anthracnose Leaf):")
    # Allow sprayer cooldown to elapse
    time.sleep(4.2)
    with open("test_samples/sample_anthracnose.jpg", "rb") as f:
        anth_b64 = base64.b64encode(f.read()).decode('utf-8')

    res_live_leaf = post_json("/api/predict_live", {"image": anth_b64})
    print(f"   Prediction: {res_live_leaf['prediction']} ({res_live_leaf['confidence']}%)")
    print(f"   Leaf Verified: {res_live_leaf['is_leaf']}")
    print(f"   Auto-Spray Triggered: {res_live_leaf['auto_spray_trigger']['triggered']} -> PASS")
    assert res_live_leaf["is_leaf"] is True
    assert res_live_leaf["auto_spray_trigger"]["triggered"] is True

    # 6. Continuous Live Frame Detection (Non-Leaf Rejection)
    print("\n6. Testing Live Frame Strict Non-Leaf Rejection (Human Face):")
    with open("test_samples/sample_nonleaf_face.jpg", "rb") as f:
        face_b64 = base64.b64encode(f.read()).decode('utf-8')

    res_live_face = post_json("/api/predict_live", {"image": face_b64})
    print(f"   Rejected Message: \"{res_live_face['message']}\"")
    print(f"   is_leaf = {res_live_face['is_leaf']} -> PASS")
    assert res_live_face["is_leaf"] is False
    assert res_live_face["message"] == "Invalid Image! Please upload a clear plant leaf image."

    print("\n" + "=" * 65)
    print("  ALL SMART ROVER & CONTINUOUS AI TESTS PASSED SUCCESSFULLY! 🎉")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_tests()
