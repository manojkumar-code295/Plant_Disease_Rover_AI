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
        """Checks if the image contains any human face."""
        if self.face_cascade is None or self.face_cascade.empty():
            return False
        # Convert PIL to OpenCV grayscale
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        return len(faces) > 0

    def verify_leaf(self, pil_image):
        """
        Runs the complete multi-tier leaf verification pipeline.
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

        # Tier 1: Strict Human Face Detection
        if self.check_face(pil_image):
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
            leaf_idx = self.class_to_idx.get("leaf", 1)
            nonleaf_idx = self.class_to_idx.get("non_leaf", 0)

            leaf_prob = float(probs[leaf_idx].item())
            nonleaf_prob = float(probs[nonleaf_idx].item())

        # If the verifier CNN indicates non-leaf or leaf probability is below threshold
        if leaf_prob < 0.70 or nonleaf_prob > 0.30:
            return {
                "is_leaf": False,
                "confidence": round(leaf_prob * 100, 2),
                "message": INVALID_MESSAGE,
                "details": f"CNN verifier classified as non-leaf (Leaf Prob: {leaf_prob*100:.1f}%, Non-Leaf Prob: {nonleaf_prob*100:.1f}%)."
            }

        # Tier 3: Passed all checks
        return {
            "is_leaf": True,
            "confidence": round(leaf_prob * 100, 2),
            "message": "Leaf verified successfully.",
            "details": f"Leaf confidence: {leaf_prob*100:.2f}%"
        }
