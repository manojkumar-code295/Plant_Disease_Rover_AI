"""
Live Terminal Logger Module.
Outputs real-time detection, motor control, and water pump status directly
into the VS Code terminal without duplicate flooding.
"""

import time
import os
from src.config import SYSTEM_LOG_PATH

class TerminalLogger:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TerminalLogger()
        return cls._instance

    def __init__(self):
        self.last_log_state = None
        self.last_log_time = 0
        self.repeat_interval = 3.0  # seconds to wait before repeating identical state

        # Initialize system log file
        os.makedirs(os.path.dirname(SYSTEM_LOG_PATH), exist_ok=True)
        if not os.path.exists(SYSTEM_LOG_PATH):
            with open(SYSTEM_LOG_PATH, "w") as f:
                f.write(f"=== AgriVision AI System Log Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    def _write_file_log(self, text):
        try:
            with open(SYSTEM_LOG_PATH, "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
        except Exception:
            pass

    def log_camera_status(self, connected=True):
        msg = "[CAMERA] Webcam Connected" if connected else "[CAMERA] Webcam Disconnected"
        print(f"\033[96m{msg}\033[0m")
        self._write_file_log(msg)

    def log_detection(self, obj_info, leaf_verified, disease_result, motor_telemetry, pump_telemetry, obstacle_info=None):
        """
        Prints detection information formatted as required:
        [OBJECT DETECTION] Leaf Detected / No Leaf Detected / Obstacle Detected
        [LEAF VERIFICATION] Valid Plant Leaf / Non-Leaf Image
        [DISEASE DETECTION] ...
        [CONFIDENCE] ...
        [MOTOR CONTROL] ...
        [WATER PUMP] ...
        [SPRAYING] ...
        """
        now = time.time()
        
        # Determine state signature to prevent duplicate terminal spam
        is_leaf = leaf_verified and disease_result and disease_result.get("is_leaf", False)
        disease_name = (disease_result.get("predicted_class") or disease_result.get("prediction", "Unknown")) if is_leaf else "None"
        conf = disease_result.get("confidence", 0) if is_leaf else 0
        has_obstacle = obstacle_info and obstacle_info.get("detected", False)
        pump_active = pump_telemetry.get("pump_active", False)
        motor_dir = motor_telemetry.get("direction", "STOP")
        motor_speed = motor_telemetry.get("speed", 65)

        current_state = (is_leaf, disease_name, int(conf // 5), has_obstacle, pump_active, motor_dir, motor_speed)

        # Suppress duplicate message if unchanged and within repeat_interval
        if self.last_log_state == current_state and (now - self.last_log_time) < self.repeat_interval:
            return

        self.last_log_state = current_state
        self.last_log_time = now

        timestamp = time.strftime("%H:%M:%S")
        lines = []

        if has_obstacle:
            obs_label = obstacle_info.get("label", "Obstacle")
            lines.append(f"\033[91m[{timestamp}] [OBJECT DETECTION] Obstacle Detected: {obs_label}\033[0m")
            lines.append(f"\033[93m[{timestamp}] [MOTOR CONTROL] Obstacle in path -> Stopped!\033[0m")

        if is_leaf:
            lines.append(f"\033[92m[{timestamp}] [OBJECT DETECTION] Leaf Detected\033[0m")
            lines.append(f"\033[92m[{timestamp}] [LEAF VERIFICATION] Valid Plant Leaf\033[0m")
            lines.append(f"\033[94m[{timestamp}] [DISEASE DETECTION] {disease_name} Detected\033[0m")
            lines.append(f"\033[94m[{timestamp}] [CONFIDENCE] {conf:.1f}%\033[0m")
            
            if pump_active:
                lines.append(f"\033[96m[{timestamp}] [WATER PUMP] Activated (Water Only)\033[0m")
                lines.append(f"\033[96m[{timestamp}] [SPRAYING] Started\033[0m")
            else:
                lines.append(f"[{timestamp}] [WATER PUMP] Standby")

            lines.append(f"[{timestamp}] [MOTOR CONTROL] {motor_dir} ({motor_speed}% Speed)")

        elif not has_obstacle:
            lines.append(f"[{timestamp}] [OBJECT DETECTION] No Leaf Detected")
            lines.append(f"[{timestamp}] [WATER PUMP] OFF")
            lines.append(f"[{timestamp}] [MOTOR CONTROL] Standby")

        # Print to terminal
        for line in lines:
            print(line)
            # Remove ANSI colors for file logging
            clean_text = line.replace('\033[91m','').replace('\033[92m','').replace('\033[93m','').replace('\033[94m','').replace('\033[96m','').replace('\033[0m','')
            self._write_file_log(clean_text)

terminal_logger = TerminalLogger.get_instance()
