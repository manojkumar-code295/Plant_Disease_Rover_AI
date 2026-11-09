#!/usr/bin/env python3
"""
AgriVision AI - Autonomous Raspberry Pi Edge Agent & Rover Controller.
Connects camera, executes dual-model inference pipeline (YOLO + Leaf Verifier + Disease CNN),
actuates differential drive motors, controls precision water spraying (WATER ONLY),
logs to the terminal in real time with duplicate suppression, archives images locally,
and syncs bi-directionally with the cloud web service on Render.
"""

import os
import sys
import time
import signal
import threading
import cv2
import requests

# Set project root in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import PlantDiseaseDetectionPipeline
from src.hardware_controller import hardware_controller
from src.terminal_logger import terminal_logger
from src.image_logger import image_logger
import src.config as config

# Configuration
RENDER_URL = os.environ.get("RENDER_URL", "https://precision-agriculture-smart-rover.onrender.com").rstrip("/")
LOCAL_URL = os.environ.get("LOCAL_URL", "http://127.0.0.1:5000").rstrip("/")
API_BASE_URL = os.environ.get("API_BASE_URL", RENDER_URL)
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", 0.5))
HEARTBEAT_INTERVAL = float(os.environ.get("HEARTBEAT_INTERVAL", 4.0))

class PiEdgeAgent:
    def __init__(self, camera_index=0, api_url=API_BASE_URL):
        self.camera_index = camera_index
        self.api_url = api_url
        self.pipeline = PlantDiseaseDetectionPipeline()
        self.hardware = hardware_controller
        self.term_logger = terminal_logger
        self.img_logger = image_logger
        self.running = False
        self.cap = None
        self.last_heartbeat = 0
        self.last_poll = 0
        self.last_uploaded_time = 0

    def start_camera(self):
        """Initializes camera feed (OpenCV or Picamera2 fallback)."""
        self.term_logger.log_camera_status(False)
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            # Try camera index 1 if 0 fails
            self.cap = cv2.VideoCapture(1)
        
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.term_logger.log_camera_status(True)
            return True
        else:
            print("[CAMERA] Error: Could not connect to camera device.")
            return False

    def sync_cloud_commands(self):
        """Polls cloud or local server for manual navigation / spray commands."""
        now = time.time()
        if now - self.last_poll < POLL_INTERVAL:
            return

        self.last_poll = now
        try:
            resp = requests.get(f"{self.api_url}/api/rover/command", timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                cmd = data.get("command")
                if cmd and cmd != "none":
                    if cmd == "forward":
                        self.hardware.forward()
                    elif cmd == "backward":
                        self.hardware.backward()
                    elif cmd == "left":
                        self.hardware.turn_left()
                    elif cmd == "right":
                        self.hardware.turn_right()
                    elif cmd == "stop":
                        self.hardware.stop()
                    elif cmd == "pump_on":
                        self.hardware.pump_on()
                    elif cmd == "pump_off":
                        self.hardware.pump_off()
                    elif cmd == "burst":
                        self.hardware.trigger_spray_burst(self.hardware.spray_duration)
                
                speed = data.get("speed")
                if speed is not None and speed != self.hardware.speed:
                    self.hardware.set_speed(speed)
        except Exception:
            pass  # Fail gracefully without blocking edge loop

    def send_heartbeat(self):
        """Sends rover heartbeat & telemetry to the cloud dashboard."""
        now = time.time()
        if now - self.last_heartbeat < HEARTBEAT_INTERVAL:
            return

        self.last_heartbeat = now
        try:
            payload = {
                "telemetry": self.hardware.get_telemetry(),
                "timestamp": time.time()
            }
            requests.post(f"{self.api_url}/api/rover/heartbeat", json=payload, timeout=2.0)
        except Exception:
            pass

    def upload_result_to_cloud(self, result_dict, frame_bgr):
        """Uploads confirmed leaf detection diagnostic to Render cloud dashboard."""
        now = time.time()
        if now - self.last_uploaded_time < 2.5:
            return  # Rate limit uploads to cloud

        self.last_uploaded_time = now
        try:
            _, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            files = {"file": ("leaf.jpg", buffer.tobytes(), "image/jpeg")}
            data = {
                "disease": result_dict.get("predicted_class", "Unknown"),
                "confidence": str(round(result_dict.get("confidence", 0.0), 1)),
                "is_leaf": str(result_dict.get("is_leaf", False)),
                "treatment": result_dict.get("treatment_notes", "")
            }
            requests.post(f"{self.api_url}/api/update-result", data=data, files=files, timeout=3.0)
        except Exception:
            pass

    def run(self):
        """Main real-time detection & rover control loop."""
        self.running = True
        if not self.start_camera():
            print("[CAMERA] Running in test mode with synthetic camera stream.")

        print("\n" + "=" * 65)
        print("🚀 AgriVision AI Edge Agent Running")
        print(f"   Platform Mode: {self.hardware.hardware_mode}")
        print(f"   Cloud Bridge URL: {self.api_url}")
        print("   Live terminal telemetry active below (Ctrl+C to stop)...")
        print("=" * 65 + "\n")

        frame_count = 0

        try:
            while self.running:
                frame_count += 1

                # 1. Grab camera frame
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.03)
                        continue
                else:
                    # Synthetic / test frame placeholder
                    import numpy as np
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    time.sleep(0.05)

                # 2. Process through full AI pipeline every 2 frames for optimal FPS
                if frame_count % 2 == 0:
                    pipeline_result = self.pipeline.process_frame(frame)
                    
                    # 3. Synchronize detection diagnostics to cloud if valid leaf was analyzed
                    leaf_res = pipeline_result.get("leaf_disease")
                    if leaf_res and leaf_res.get("is_leaf"):
                        threading.Thread(
                            target=self.upload_result_to_cloud,
                            args=(leaf_res, frame),
                            daemon=True
                        ).start()

                # 4. Synchronize remote commands and send heartbeat
                self.sync_cloud_commands()
                self.send_heartbeat()

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n[SYSTEM] Stopping AgriVision AI Edge Agent...")
        finally:
            self.stop()

    def stop(self):
        """Cleans up camera, stops motors, turns off pump safely."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.hardware.emergency_stop()
        self.term_logger.log_camera_status(False)
        print("[SYSTEM] Clean shutdown complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AgriVision AI Edge Agent")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--url", type=str, default=API_BASE_URL, help="Target Cloud/Local API URL")
    args = parser.parse_args()

    agent = PiEdgeAgent(camera_index=args.camera, api_url=args.url)
    agent.run()
