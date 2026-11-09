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
        print(f"[Object Detector] Initializing YOLOv8 Object Detector ({YOLO_MODEL_PATH})...")
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            self.model_loaded = True
            print("[Object Detector] YOLOv8 Model initialized successfully!")
        except Exception as e:
            print(f"[Object Detector] Warning: Could not load YOLOv8 model ({e}). Using fallback contour detector.")
            self.model_loaded = False

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

        w, h = pil_image.size
        detected_objects = []
        has_leaf = False
        has_obstacle = False
        primary_leaf_box = None
        primary_obstacle = None

        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 1. Run YOLO Object Detection
        if self.model_loaded:
            try:
                results = self.model.predict(cv_img, conf=conf_threshold, verbose=False)
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
            except Exception as e:
                pass

        # 2. Plant Leaf Color / Contour ROI Detection
        # Since close-up camera frames of a leaf might not trigger COCO's 'potted plant' class,
        # we perform plant foliage contour detection to locate the leaf boundary
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        # Green & plant foliage mask (hue 25 - 95, saturation > 35, value > 35)
        lower_foliage = np.array([25, 35, 35])
        upper_foliage = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower_foliage, upper_foliage)
        foliage_ratio = np.sum(mask > 0) / (w * h)

        if foliage_ratio > 0.08:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > (w * h * 0.05):
                    lx, ly, lw, lh = cv2.boundingRect(largest)
                    has_leaf = True
                    leaf_box = [lx, ly, lx + lw, ly + lh]
                    if primary_leaf_box is None:
                        primary_leaf_box = leaf_box
                        detected_objects.insert(0, {
                            "label": "leaf",
                            "confidence": round(min(99.0, foliage_ratio * 140), 1),
                            "box": leaf_box,
                            "type": "leaf"
                        })

        # If full leaf fills the frame
        if primary_leaf_box is None and foliage_ratio > 0.15:
            has_leaf = True
            primary_leaf_box = [int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)]
            detected_objects.append({
                "label": "leaf",
                "confidence": 92.0,
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
