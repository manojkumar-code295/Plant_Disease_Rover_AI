"""
Disease Classifier Module.
Evaluates verified leaf images using the trained CNN model.
Classifies into exactly 4 categories:
  1. Healthy Leaf
  2. Cercospora
  3. Anthracnose
  4. Bacterial Spot
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "plant_disease_cnn.pth")
CLASSES_PATH = os.path.join(MODELS_DIR, "disease_classes.json")

CONFIDENCE_THRESHOLD = 0.70  # 70% threshold for reliable detection

# Disease Knowledge Base
DISEASE_KNOWLEDGE = {
    "Healthy": {
        "display_name": "Healthy Leaf",
        "status": "Healthy Plant",
        "is_healthy": True,
        "badge_color": "#10b981",  # Emerald Green
        "description": "The leaf shows vigorous chlorophyll pigmentation, smooth intact surfaces, and no visible signs of microbial or fungal pathogen lesions.",
        "pathogen": "None (Plant is in optimal physiological health)",
        "symptoms": [
            "Even green pigmentation across the leaf lamina",
            "Clear, unobstructed venation patterns",
            "Absence of necrotic lesions, yellow halos, or fungal spore pustules"
        ],
        "treatments": {
            "preventive": "Continue balanced watering (avoid waterlogging or drought stress).",
            "soil": "Maintain balanced N-P-K nutrient feeding and routine soil organic matter enrichment.",
            "monitoring": "Conduct regular weekly visual leaf scouting for early pest or pathogen indicators."
        }
    },
    "Cercospora": {
        "display_name": "Cercospora Leaf Spot",
        "status": "Diseased (Fungal Infection)",
        "is_healthy": False,
        "badge_color": "#f59e0b",  # Amber
        "description": "Cercospora leaf spot (e.g., Gray Leaf Spot) is a widespread fungal disease caused by species of Cercospora, thriving in humid conditions and prolonged leaf wetness.",
        "pathogen": "Fungus (Cercospora zeae-maydis / Cercospora spp.)",
        "symptoms": [
            "Small, tan spots with chlorotic yellow borders",
            "Lesions expand into elongated rectangular gray-to-brown necrotic lesions bounded by leaf veins",
            "Extensive blighting of the leaf canopy in severe infestations"
        ],
        "treatments": {
            "chemical": "Foliar fungicides such as Azoxystrobin, Pyraclostrobin, or Propiconazole applied at early onset.",
            "organic": "Liquid copper fungicides, Neem seed extract foliar spray, or bio-fungicide Bacillus amyloliquefaciens.",
            "prevention": "Rotate crops with non-susceptible species; bury or compost infected post-harvest debris; improve plant spacing to optimize airflow."
        }
    },
    "Anthracnose": {
        "display_name": "Anthracnose Disease",
        "status": "Diseased (Fungal Infection)",
        "is_healthy": False,
        "badge_color": "#ef4444",  # Red
        "description": "Anthracnose is a destructive fungal infection caused by Colletotrichum species, causing sunken, dark necrotic lesions across leaves, stems, and blossoms.",
        "pathogen": "Fungus (Colletotrichum gloeosporioides / Colletotrichum spp.)",
        "symptoms": [
            "Sunken, dark brown to black circular or irregular leaf spots",
            "Concentric rings or shot-hole centers within necrotic areas",
            "Premature defoliation, twig dieback, and blighted branch tips"
        ],
        "treatments": {
            "chemical": "Fungicides containing Chlorothalonil, Mancozeb, or systemic Copper Hydroxide.",
            "organic": "Bordeaux mixture (copper sulfate + slaked lime), Bacillus subtilis bio-fungicide, sulfur spray.",
            "prevention": "Prune interior branches to increase sunlight and rapid leaf drying; sanitize pruning equipment between cuts; avoid overhead irrigation."
        }
    },
    "Bacterial_Spot": {
        "display_name": "Bacterial Spot",
        "status": "Diseased (Bacterial Infection)",
        "is_healthy": False,
        "badge_color": "#ec4899",  # Rose/Pink
        "description": "Bacterial spot is caused by Xanthomonas species. It attacks both foliage and fruit in high-humidity warm environments, entering through natural pores (stomata) and wounds.",
        "pathogen": "Bacterium (Xanthomonas campestris / Xanthomonas vesicatoria)",
        "symptoms": [
            "Small, dark, water-soaked angular leaf spots usually surrounded by bright yellow halos",
            "Dead tissue centers may drop out, creating a shot-hole appearance",
            "Leaves turn chlorotic yellow and drop prematurely, exposing foliage to sunscald"
        ],
        "treatments": {
            "chemical": "Copper-based bactericides combined with Mancozeb (to mitigate copper-resistant bacterial strains).",
            "organic": "Copper octanoate (copper soap) and bio-bactericides containing Bacillus amyloliquefaciens strain D747.",
            "prevention": "Use certified disease-free seeds; do not handle foliage when wet; implement a 2-3 year crop rotation away from solanaceous plants."
        }
    }
}

class DiseaseClassifier:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(CLASSES_PATH, "r") as f:
            classes_info = json.load(f)
            self.class_to_idx = classes_info["class_to_idx"]
            self.idx_to_class = {int(k): v for k, v in classes_info["idx_to_class"].items()}

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
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, len(self.class_to_idx))
        )
        checkpoint = torch.load(WEIGHTS_PATH, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def predict(self, pil_image):
        """
        Classifies the verified leaf image into one of 4 categories.
        Returns:
            dict with classification result, confidence, probabilities, and disease knowledge.
        """
        tensor = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        best_idx = int(np.argmax(probs))
        best_class = self.idx_to_class[best_idx]
        best_conf = float(probs[best_idx])

        # All class breakdown
        all_probs = {self.idx_to_class[i]: round(float(probs[i]) * 100, 2) for i in range(len(probs))}

        # Confidence threshold check
        if best_conf < CONFIDENCE_THRESHOLD:
            return {
                "status": "uncertain",
                "message": "Uncertain Prediction. Please upload another clear leaf image.",
                "predicted_class": best_class,
                "confidence": round(best_conf * 100, 2),
                "threshold": round(CONFIDENCE_THRESHOLD * 100, 2),
                "all_probabilities": all_probs,
                "disease_info": None
            }

        info = DISEASE_KNOWLEDGE.get(best_class, {})

        return {
            "status": "success",
            "message": f"Successfully classified as {info.get('display_name', best_class)}.",
            "predicted_class": best_class,
            "display_name": info.get("display_name", best_class),
            "confidence": round(best_conf * 100, 2),
            "is_healthy": info.get("is_healthy", False),
            "health_status": info.get("status", "Unknown"),
            "all_probabilities": all_probs,
            "disease_info": info
        }
