"""
Automatic Image Storage and CSV History Logging Module.
Automatically archives detected leaf images into categorized folders:
  detected_images/Healthy/
  detected_images/Cercospora/
  detected_images/Anthracnose/
  detected_images/Bacterial_Spot/

Appends structured records to detection_logs/detection_history.csv.
Includes duplicate suppression to prevent excessive image flooding.
"""

import os
import csv
import time
from PIL import Image
from src.config import (
    DETECTED_IMAGES_DIR,
    DETECTION_LOGS_DIR,
    CSV_LOG_PATH,
    MIN_IMAGE_CAPTURE_INTERVAL,
    SAVE_HEALTHY_IMAGES,
    SAVE_DISEASED_IMAGES
)

class ImageLogger:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ImageLogger()
        return cls._instance

    def __init__(self):
        self.last_saved_time = {}
        self.min_interval = MIN_IMAGE_CAPTURE_INTERVAL
        self.storage_enabled = True
        self.save_healthy = SAVE_HEALTHY_IMAGES
        self.save_diseased = SAVE_DISEASED_IMAGES

        # Counters
        self.total_leaves_detected = 0
        self.total_healthy_leaves = 0
        self.total_diseased_leaves = 0
        self.total_images_stored = 0

        self._init_csv()

    def _init_csv(self):
        os.makedirs(DETECTION_LOGS_DIR, exist_ok=True)
        if not os.path.exists(CSV_LOG_PATH):
            with open(CSV_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Filename",
                    "Timestamp",
                    "Disease_Classification",
                    "Confidence_Percent",
                    "Detection_Status",
                    "Spraying_Status",
                    "Motor_Status",
                    "Motor_Speed_PWM"
                ])

    def save_detection(self, pil_image, detection_result, motor_telemetry, pump_telemetry):
        """
        Saves image and appends CSV record if criteria are met.
        Returns dict with status and image path.
        """
        if not self.storage_enabled or pil_image is None or not detection_result:
            return {"saved": False, "reason": "disabled_or_none"}

        is_leaf = detection_result.get("is_leaf", False)
        if not is_leaf:
            return {"saved": False, "reason": "non_leaf"}

        disease = detection_result.get("predicted_class") or detection_result.get("prediction", "Unknown")
        is_healthy = detection_result.get("is_healthy", False)
        conf = detection_result.get("confidence", 0.0)

        # Update lifetime counters
        self.total_leaves_detected += 1
        if is_healthy:
            self.total_healthy_leaves += 1
        else:
            self.total_diseased_leaves += 1

        # Check filter preferences
        if is_healthy and not self.save_healthy:
            return {"saved": False, "reason": "save_healthy_disabled"}
        if not is_healthy and not self.save_diseased:
            return {"saved": False, "reason": "save_diseased_disabled"}

        # Duplicate suppression: ensure min_interval seconds elapsed for this category
        now = time.time()
        last_time = self.last_saved_time.get(disease, 0)
        if (now - last_time) < self.min_interval:
            return {"saved": False, "reason": "duplicate_suppressed"}

        self.last_saved_time[disease] = now

        # Target directory: detected_images/<disease>/
        target_dir = os.path.join(DETECTED_IMAGES_DIR, disease)
        os.makedirs(target_dir, exist_ok=True)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"leaf_{disease}_{timestamp_str}_{int(conf)}pct.jpg"
        save_path = os.path.join(target_dir, filename)

        try:
            pil_image.save(save_path, "JPEG", quality=90)
            self.total_images_stored += 1

            # Append to CSV
            pump_status = "ACTIVE_SPRAYING" if pump_telemetry.get("pump_active") else "OFF"
            motor_status = motor_telemetry.get("direction", "STOP")
            motor_speed = motor_telemetry.get("speed", 65)
            dt_str = time.strftime("%Y-%m-%d %H:%M:%S")

            with open(CSV_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    filename,
                    dt_str,
                    disease,
                    f"{conf:.1f}%",
                    "Valid Leaf Verified",
                    pump_status,
                    motor_status,
                    f"{motor_speed}%"
                ])

            return {
                "saved": True,
                "filename": filename,
                "path": save_path,
                "disease": disease,
                "total_stored": self.total_images_stored
            }

        except Exception as e:
            return {"saved": False, "error": str(e)}

    def get_stats(self):
        """Returns statistics for website live dashboard."""
        return {
            "total_leaves_detected": self.total_leaves_detected,
            "total_healthy_leaves": self.total_healthy_leaves,
            "total_diseased_leaves": self.total_diseased_leaves,
            "total_images_stored": self.total_images_stored
        }

image_logger = ImageLogger.get_instance()
