"""
Leaf Verifier Module.
Strictly ensures that an uploaded or captured image is a genuine plant leaf.
Rejects:
  - Human faces / bodies
  - Animals
  - Vehicles
  - Buildings
  - Fruits
  - Flowers
  - Soil
  - Trees without identifiable leaves
  - Random objects
  - Screenshots / drawings / unrelated images

Exact required rejection message:
  "Invalid Image! Please upload a clear plant leaf image."
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
HAAR_FACE_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
VERIFIER_WEIGHTS_PATH = os.path.join(MODELS_DIR, "leaf_verifier_cnn.pth")
VERIFIER_CLASSES_PATH = os.path.join(MODELS_DIR, "verifier_classes.json")

INVALID_MESSAGE = "Invalid Image! Please upload a clear plant leaf image."

class LeafVerifier:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize Face Cascade
        self.face_cascade = None
        if os.path.exists(HAAR_FACE_PATH):
            self.face_cascade = cv2.CascadeClassifier(HAAR_FACE_PATH)

        # 2. Load Verifier Classes
        with open(VERIFIER_CLASSES_PATH, "r") as f:
            classes_info = json.load(f)
            self.class_to_idx = classes_info["class_to_idx"]
            self.idx_to_class = {int(k): v for k, v in classes_info["idx_to_class"].items()}

        # 3. Load Verifier CNN
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )
        checkpoint = torch.load(VERIFIER_WEIGHTS_PATH, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def check_face(self, pil_image):
        """
        Checks if the image contains any real human face using skin chrominance verification.
        Eliminates false face detections caused by leaf veins, spots, or natural foliage shadows.
        """
        if self.face_cascade is None or self.face_cascade.empty():
            return False, []
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
        real_faces = []
        for (fx, fy, fw, fh) in faces:
            roi = cv_img[fy:fy+fh, fx:fx+fw]
            if roi.size == 0:
                continue
            # Human skin detection in YCrCb color space
            ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
            cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
            skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
            skin_pct = np.mean(skin_mask) * 100

            # Plant chlorophyll / foliage detection in HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            green_mask = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 25)
            green_pct = np.mean(green_mask) * 100

            # A genuine human face has substantial human skin and lack of plant chlorophyll
            if skin_pct >= 18.0 and green_pct < 25.0:
                real_faces.append([int(fx), int(fy), int(fx + fw), int(fy + fh)])

        return len(real_faces) > 0, real_faces

    def verify_leaf(self, pil_image):
        """
        Runs the complete botanical leaf verification pipeline.
        Returns:
            dict: {
                "is_leaf": bool,
                "confidence": float,
                "message": str,
                "details": str
            }
        """
        # Tier 0: Image integrity and size check
        if pil_image is None or pil_image.size[0] < 30 or pil_image.size[1] < 30:
            return {
                "is_leaf": False,
                "confidence": 0.0,
                "message": INVALID_MESSAGE,
                "details": "Image resolution too low or corrupt."
            }

        pil_image = pil_image.convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        h, w = cv_img.shape[:2]

        # Tier 1: Real Human Face Detection (Skin-chrominance validated)
        has_face, real_faces = self.check_face(pil_image)

        # Botanical vegetation analysis: Excess Green Index (ExG)
        rgb_f = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
        exg = 2.0 * g - r - b
        foliage_ratio = float(np.mean(exg > 0.03))

        # Also check HSV leaf pigment (chlorophyll, yellow chlorosis, brown necrotic lesions)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        leaf_pigment_mask = (h_ch >= 15) & (h_ch <= 100) & (s_ch >= 20) & (v_ch >= 20)
        pigment_ratio = float(np.mean(leaf_pigment_mask))

        # If a real human face dominates the image and no plant foliage exists (pure selfie)
        if has_face and foliage_ratio < 0.04 and pigment_ratio < 0.08:
            return {
                "is_leaf": False,
                "confidence": 0.0,
                "message": INVALID_MESSAGE,
                "details": "Human face detected."
            }

        # Tier 2: Dedicated Leaf Verifier CNN
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            leaf_idx = self.class_to_idx.get("leaf", 0)
            nonleaf_idx = self.class_to_idx.get("non_leaf", 1)

            leaf_prob = float(probs[leaf_idx].item())
            nonleaf_prob = float(probs[nonleaf_idx].item())
        del tensor, outputs, probs

        # Reject confirmed non-leaf objects (cars, buildings, animals, random objects)
        if nonleaf_prob > 0.85 and foliage_ratio < 0.15 and pigment_ratio < 0.20:
            return {
                "is_leaf": False,
                "confidence": round(leaf_prob * 100, 2),
                "message": INVALID_MESSAGE,
                "details": f"CNN verifier classified as non-leaf (Leaf Prob: {leaf_prob*100:.1f}%, Non-Leaf Prob: {nonleaf_prob*100:.1f}%)."
            }

        # Severe non-leaf certainty
        if nonleaf_prob > 0.96 and leaf_prob < 0.04:
            return {
                "is_leaf": False,
                "confidence": round(leaf_prob * 100, 2),
                "message": INVALID_MESSAGE,
                "details": f"CNN verifier classified as non-leaf (Leaf Prob: {leaf_prob*100:.1f}%, Non-Leaf Prob: {nonleaf_prob*100:.1f}%)."
            }

        # Valid leaf confirmation: passes if CNN confirms or botanical vegetation index confirms leaf tissue
        if leaf_prob >= 0.35 or foliage_ratio >= 0.04 or pigment_ratio >= 0.10:
            computed_conf = max(leaf_prob * 100, min(99.9, max(foliage_ratio, pigment_ratio) * 150))
            return {
                "is_leaf": True,
                "confidence": round(computed_conf, 2),
                "message": "Leaf verified successfully.",
                "details": f"Leaf confidence: {computed_conf:.2f}%"
            }

        return {
            "is_leaf": False,
            "confidence": round(leaf_prob * 100, 2),
            "message": INVALID_MESSAGE,
            "details": "No verified plant leaf foliage detected."
        }
