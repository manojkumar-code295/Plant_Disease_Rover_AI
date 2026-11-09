"""
Live Object Detection Module using YOLOv8.
Detects:
  - Plant leaves & potted plants
  - Obstacles in front of the rover (people, animals, vehicles, furniture, everyday items)
  - Distinguishes leaves from unrelated objects
  - Returns normalized and pixel bounding boxes with labels and confidence scores
"""

import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from src.config import OBSTACLE_CLASSES, YOLO_MODEL_PATH

class LiveObjectDetector:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LiveObjectDetector()
        return cls._instance

    def __init__(self):
        self.model = None
        self.model_loaded = False

    def load_model(self):
        if self.model_loaded and self.model is not None:
            return True
        print(f"[Object Detector] Initializing YOLOv8 Object Detector ({YOLO_MODEL_PATH})...")
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            self.model_loaded = True
            print("[Object Detector] YOLOv8 Model initialized successfully!")
            return True
        except Exception as e:
            print(f"[Object Detector] Warning: Could not load YOLOv8 model ({e}). Using fallback contour detector.")
            self.model_loaded = False
            return False

    def detect(self, pil_image, conf_threshold=0.35):
        """
        Runs object detection on the image.
        Returns:
            dict with:
              - detected_objects: list of dicts with label, conf, box [x1, y1, x2, y2], type ('leaf' or 'obstacle')
              - has_leaf: bool
              - has_obstacle: bool
              - primary_leaf_box: [x1, y1, x2, y2]
              - primary_obstacle: dict or None
        """
        if pil_image is None:
            return {"detected_objects": [], "has_leaf": False, "has_obstacle": False, "primary_leaf_box": None, "primary_obstacle": None}

        if not self.model_loaded and self.model is None:
            self.load_model()

        w, h = pil_image.size
        detected_objects = []
        has_leaf = False
        has_obstacle = False
        primary_leaf_box = None
        primary_obstacle = None

        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 1. Run YOLO Object Detection for Obstacles / Surroundings
        if self.model_loaded and self.model is not None:
            try:
                results = self.model.predict(cv_img, imgsz=320, conf=conf_threshold, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = self.model.names[cls_id]
                        score = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        x1, y1, x2, y2 = xyxy

                        # Determine if obstacle or plant
                        if cls_name in OBSTACLE_CLASSES:
                            has_obstacle = True
                            obj_info = {
                                "label": cls_name,
                                "confidence": round(score * 100, 1),
                                "box": [x1, y1, x2, y2],
                                "type": "obstacle"
                            }
                            detected_objects.append(obj_info)
                            if primary_obstacle is None or score > primary_obstacle["confidence"]:
                                primary_obstacle = obj_info
                        elif cls_name in ["potted plant"]:
                            has_leaf = True
                            obj_info = {
                                "label": "plant",
                                "confidence": round(score * 100, 1),
                                "box": [x1, y1, x2, y2],
                                "type": "leaf"
                            }
                            detected_objects.append(obj_info)
                            if primary_leaf_box is None:
                                primary_leaf_box = [x1, y1, x2, y2]
                del results
            except Exception as e:
                pass

        # 2. Plant Leaf Botanical Saliency & Contour Localization
        # Computes Excess Green Index (ExG = 2G - R - B) and foliage color features
        rgb_f = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
        exg = 2.0 * g - r - b

        # HSV foliage mask (covers chlorophyll green, chlorosis yellow, and lesion margins)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        foliage_mask = (h_ch >= 15) & (h_ch <= 100) & (s_ch >= 20) & (v_ch >= 20)

        # Combined botanical presence mask
        combined_leaf_mask = ((exg > 0.03) | foliage_mask).astype(np.uint8) * 255
        foliage_ratio = float(np.mean(combined_leaf_mask > 0))

        # Morphological closing to consolidate leaf blade
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        clean_mask = cv2.morphologyEx(combined_leaf_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []

        if contours:
            for c in contours:
                area = cv2.contourArea(c)
                # Leaf region must occupy at least 1.5% of the frame
                if area > (w * h * 0.015):
                    bx, by, bw, bh = cv2.boundingRect(c)
                    # 5% safety padding
                    px = int(bw * 0.05)
                    py = int(bh * 0.05)
                    x1 = max(0, bx - px)
                    y1 = max(0, by - py)
                    x2 = min(w, bx + bw + px)
                    y2 = min(h, by + bh + py)
                    candidate_boxes.append((area, [x1, y1, x2, y2]))

        candidate_boxes.sort(key=lambda item: item[0], reverse=True)

        # If localized leaf contours exist
        if candidate_boxes:
            has_leaf = True
            for area, box in candidate_boxes[:3]:
                conf_score = round(min(99.8, max(75.0, (area / (w * h)) * 250)), 1)
                leaf_obj = {
                    "label": "leaf",
                    "confidence": conf_score,
                    "box": box,
                    "type": "leaf"
                }
                # Put primary leaf at the top
                if primary_leaf_box is None:
                    primary_leaf_box = box
                    detected_objects.insert(0, leaf_obj)
                else:
                    detected_objects.append(leaf_obj)

        # If no localized contour but strong global foliage exists (e.g., leaf close-up fills camera view)
        elif foliage_ratio > 0.10:
            has_leaf = True
            primary_leaf_box = [int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)]
            detected_objects.insert(0, {
                "label": "leaf",
                "confidence": round(min(99.0, max(75.0, foliage_ratio * 120)), 1),
                "box": primary_leaf_box,
                "type": "leaf"
            })

        return {
            "detected_objects": detected_objects,
            "has_leaf": has_leaf,
            "has_obstacle": has_obstacle,
            "primary_leaf_box": primary_leaf_box,
            "primary_obstacle": primary_obstacle
        }

object_detector = LiveObjectDetector.get_instance()
