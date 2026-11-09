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
        self.repeat_interval = 2.5  # seconds to wait before repeating identical state

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

    def log_camera_event(self, event):
        """Logs camera life-cycle events in exact requested format."""
        msg = None
        if event == "connected":
            msg = "[CAMERA] Connected"
        elif event == "video_active":
            msg = "[VIDEO] Live stream active"
        elif event == "disconnected":
            msg = "[CAMERA] Disconnected"

        if msg:
            print(f"\033[96m{msg}\033[0m")
            self._write_file_log(msg)

    def log_camera_status(self, connected=True):
        self.log_camera_event("connected" if connected else "disconnected")

    def log_detection(self, obj_info, leaf_verified, disease_result, motor_telemetry, pump_telemetry, obstacle_info=None):
        """
        Prints live detection information formatted to user requirements:
        [FRAME] Frame captured
        [AI] Frame sent for processing
        [OBJECT DETECTION] Processing started
        [LEAF DETECTION] Leaf detected / No leaf detected
        [CNN] Classification started
        [RESULT] ...
        [CONFIDENCE] ...%
        [MOTOR CONTROL] ...
        [WATER PUMP] ...
        """
        now = time.time()
        
        is_leaf = bool(leaf_verified and disease_result and disease_result.get("is_leaf", False))
        disease_name = (disease_result.get("predicted_class") or disease_result.get("prediction", "Unknown")) if is_leaf else "None"
        conf = float(disease_result.get("confidence", 0.0)) if is_leaf else 0.0
        has_obstacle = bool(obstacle_info and obstacle_info.get("detected", False))
        pump_active = bool(pump_telemetry and pump_telemetry.get("pump_active", False))
        motor_dir = motor_telemetry.get("direction", "STOP") if motor_telemetry else "STOP"
        motor_speed = motor_telemetry.get("speed", 65) if motor_telemetry else 65

        current_state = (is_leaf, disease_name, int(conf // 5), has_obstacle, pump_active, motor_dir, motor_speed)

        # Suppress duplicate logs if identical state and within repeat_interval
        if self.last_log_state == current_state and (now - self.last_log_time) < self.repeat_interval:
            return

        self.last_log_state = current_state
        self.last_log_time = now

        lines = [
            "\033[90m[FRAME] Frame captured\033[0m",
            "\033[90m[AI] Frame sent for processing\033[0m",
            "\033[94m[OBJECT DETECTION] Processing started\033[0m"
        ]

        if has_obstacle:
            obs_label = obstacle_info.get("label", "Obstacle")
            lines.append(f"\033[91m[OBJECT DETECTION] Obstacle Detected: {obs_label}\033[0m")
            lines.append(f"\033[93m[MOTOR CONTROL] Obstacle in path -> Stopped!\033[0m")

        if is_leaf:
            lines.append("\033[92m[LEAF DETECTION] Leaf detected\033[0m")
            lines.append("\033[94m[CNN] Classification started\033[0m")
            lines.append(f"\033[92m[RESULT] {disease_name}\033[0m")
            lines.append(f"\033[94m[CONFIDENCE] {conf:.1f}%\033[0m")
            
            if pump_active:
                lines.append("\033[96m[WATER PUMP] Activated (Water Only)\033[0m")
                lines.append("\033[96m[SPRAYING] Started\033[0m")
            else:
                lines.append("[WATER PUMP] Standby")

            lines.append(f"[MOTOR CONTROL] {motor_dir} ({motor_speed}% Speed)")

        elif not has_obstacle:
            lines.append("\033[93m[LEAF DETECTION] No leaf detected\033[0m")
            lines.append("[WATER PUMP] OFF")
            lines.append("[MOTOR CONTROL] Standby")

        # Print to terminal
        for line in lines:
            print(line)
            clean_text = line.replace('\033[90m','').replace('\033[91m','').replace('\033[92m','').replace('\033[93m','').replace('\033[94m','').replace('\033[96m','').replace('\033[0m','')
            self._write_file_log(clean_text)

terminal_logger = TerminalLogger.get_instance()
