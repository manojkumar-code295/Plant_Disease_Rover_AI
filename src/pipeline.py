"""
Upgraded Plant Leaf Detection and Automatic Smart Spraying Pipeline.
Integrates:
  1. Live YOLOv8 Object Detection (Leaves, Plants, and Obstacles).
  2. Strict Leaf Verifier CNN (Filters out non-leaf objects & faces).
  3. Trained Plant Disease CNN (Healthy, Cercospora, Anthracnose, Bacterial Spot).
  4. Real-time Terminal Logger (Outputs live logs to VS Code terminal).
  5. Automatic Precision Water Spray Controller (WATER ONLY with duration & cooldown).
  6. Automatic Image Storage (Archives to detected_images/<disease>/ and CSV logs).
  7. Motor Safety & Obstacle Auto-Braking.
"""

import io
import base64
from PIL import Image

from src.leaf_verifier import LeafVerifier, INVALID_MESSAGE
from src.disease_classifier import DiseaseClassifier
from src.object_detector import object_detector
from src.hardware_controller import hardware_controller
from src.terminal_logger import terminal_logger
from src.image_logger import image_logger

class PlantDiseaseDetectionPipeline:
    def __init__(self, lazy=True):
        print("Initializing AgriVision Upgraded Real-Time AI Pipeline...")
        self._lazy = lazy
        self._verifier = None
        self._classifier = None
        self._object_detector = None
        self.hardware = hardware_controller
        self.term_logger = terminal_logger
        self.img_logger = image_logger
        if not lazy:
            self._ensure_models()
        print("Pipeline components successfully integrated and ready for live execution!")

    def _ensure_models(self):
        if self._verifier is None:
            self._verifier = LeafVerifier()
        if self._classifier is None:
            self._classifier = DiseaseClassifier()
        if self._object_detector is None:
            self._object_detector = object_detector

    @property
    def verifier(self):
        if self._verifier is None:
            self._verifier = LeafVerifier()
        return self._verifier

    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = DiseaseClassifier()
        return self._classifier

    @property
    def object_detector(self):
        if self._object_detector is None:
            self._object_detector = object_detector
        return self._object_detector

    def process_image(self, pil_image, run_object_detection=True, run_disease_detection=True):
        """
        Executes the complete upgraded pipeline on incoming video frame or image.
        """
        if pil_image is None:
            return {
                "success": False,
                "is_leaf": False,
                "status": "rejected",
                "error": INVALID_MESSAGE,
                "message": INVALID_MESSAGE,
                "bounding_boxes": [],
                "hardware_telemetry": self.hardware.get_telemetry()
            }

        w, h = pil_image.size

        # -------------------------------------------------------------
        # Step 1: Object & Obstacle Detection (YOLOv8)
        # -------------------------------------------------------------
        obj_res = {"detected_objects": [], "has_leaf": False, "has_obstacle": False, "primary_leaf_box": None, "primary_obstacle": None}
        if run_object_detection:
            obj_res = self.object_detector.detect(pil_image)

        # Handle Obstacle Braking
        if obj_res["has_obstacle"] and obj_res["primary_obstacle"]:
            self.hardware.handle_obstacle_detected(obj_res["primary_obstacle"]["label"])
        elif self.hardware.obstacle_stopped and not obj_res["has_obstacle"]:
            self.hardware.handle_path_cleared()

        # -------------------------------------------------------------
        # Step 2: Leaf Verification (Strict Filter)
        # -------------------------------------------------------------
        # If object detector localized a candidate leaf ROI, we can crop and verify it
        crop_img = pil_image
        if obj_res["primary_leaf_box"]:
            bx1, by1, bx2, by2 = obj_res["primary_leaf_box"]
            # Add small padding
            pad_x = int((bx2 - bx1) * 0.05)
            pad_y = int((by2 - by1) * 0.05)
            cx1 = max(0, bx1 - pad_x)
            cy1 = max(0, by1 - pad_y)
            cx2 = min(w, bx2 + pad_x)
            cy2 = min(h, by2 + pad_y)
            if (cx2 - cx1) > 40 and (cy2 - cy1) > 40:
                crop_img = pil_image.crop((cx1, cy1, cx2, cy2))

        verifier_res = self.verifier.verify_leaf(crop_img)

        # If strict leaf check failed on candidate crop, check whole image as well
        if not verifier_res["is_leaf"] and crop_img != pil_image:
            verifier_res_full = self.verifier.verify_leaf(pil_image)
            if verifier_res_full["is_leaf"]:
                verifier_res = verifier_res_full
                crop_img = pil_image

        is_valid_leaf = verifier_res["is_leaf"]

        # -------------------------------------------------------------
        # Step 3: Disease Classification CNN
        # -------------------------------------------------------------
        clf_res = None
        if is_valid_leaf and run_disease_detection:
            clf_res = self.classifier.predict(crop_img)
            if clf_res:
                clf_res["is_leaf"] = is_valid_leaf

        # -------------------------------------------------------------
        # Step 4: Automatic Precision Water Spray Evaluation
        # -------------------------------------------------------------
        spray_trigger_info = {"triggered": False, "reason": "not_triggered"}
        if is_valid_leaf and clf_res:
            conf = clf_res.get("confidence", 0.0)
            is_healthy = clf_res.get("is_healthy", False)
            spray_trigger_info = self.hardware.evaluate_auto_spray(is_valid_leaf, is_healthy, conf)

        # -------------------------------------------------------------
        # Step 5: Automatic Image Storage & CSV History
        # -------------------------------------------------------------
        storage_info = {"saved": False}
        if is_valid_leaf and clf_res:
            storage_info = self.img_logger.save_detection(
                pil_image=crop_img,
                detection_result=clf_res,
                motor_telemetry=self.hardware.get_telemetry(),
                pump_telemetry=self.hardware.get_telemetry()
            )

        # -------------------------------------------------------------
        # Step 6: Live VS Code Terminal Output
        # -------------------------------------------------------------
        self.term_logger.log_detection(
            obj_info=obj_res,
            leaf_verified=is_valid_leaf,
            disease_result=clf_res,
            motor_telemetry=self.hardware.get_telemetry(),
            pump_telemetry=self.hardware.get_telemetry(),
            obstacle_info=obj_res["primary_obstacle"] if obj_res["has_obstacle"] else None
        )

        # -------------------------------------------------------------
        # Assemble Response
        # -------------------------------------------------------------
        telemetry = self.hardware.get_telemetry()
        stats = self.img_logger.get_stats()

        # Format Bounding Boxes for Web Canvas Overlay
        formatted_boxes = []
        for obj in obj_res["detected_objects"]:
            formatted_boxes.append({
                "label": obj["label"],
                "type": obj["type"],
                "confidence": obj["confidence"],
                "box": obj["box"]
            })

        # Clean intermediate memory
        import gc
        gc.collect()

        # If non-leaf rejected
        if not is_valid_leaf:
            return {
                "success": False,
                "is_leaf": False,
                "status": "rejected",
                "error": INVALID_MESSAGE,
                "message": INVALID_MESSAGE,
                "verifier_details": verifier_res.get("details", "Non-leaf image"),
                "bounding_boxes": formatted_boxes,
                "has_obstacle": obj_res["has_obstacle"],
                "obstacle_info": obj_res["primary_obstacle"],
                "hardware_telemetry": telemetry,
                "statistics": stats
            }

        # If uncertain prediction
        if clf_res and clf_res.get("status") == "uncertain":
            return {
                "success": True,
                "is_leaf": True,
                "status": "uncertain",
                "message": clf_res["message"],
                "leaf_confidence": verifier_res["confidence"],
                "prediction": clf_res["predicted_class"],
                "confidence": clf_res["confidence"],
                "all_probabilities": clf_res["all_probabilities"],
                "bounding_boxes": formatted_boxes,
                "has_obstacle": obj_res["has_obstacle"],
                "auto_spray_trigger": spray_trigger_info,
                "image_storage": storage_info,
                "hardware_telemetry": telemetry,
                "statistics": stats
            }

        # Successful Diagnosis
        return {
            "success": True,
            "is_leaf": True,
            "status": "success",
            "message": clf_res["message"],
            "leaf_confidence": verifier_res["confidence"],
            "prediction": clf_res["predicted_class"],
            "display_name": clf_res["display_name"],
            "confidence": clf_res["confidence"],
            "is_healthy": clf_res["is_healthy"],
            "health_status": clf_res["health_status"],
            "all_probabilities": clf_res["all_probabilities"],
            "disease_info": clf_res["disease_info"],
            "bounding_boxes": formatted_boxes,
            "has_obstacle": obj_res["has_obstacle"],
            "obstacle_info": obj_res["primary_obstacle"],
            "auto_spray_trigger": spray_trigger_info,
            "image_storage": storage_info,
            "hardware_telemetry": telemetry,
            "statistics": stats
        }
