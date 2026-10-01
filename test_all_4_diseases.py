#!/usr/bin/env python3
"""
Test script to verify all 3 diseases + Healthy Leaf and non-leaf rejection.
"""
import os
import glob
import cv2
import numpy as np

# Import detection and inference from complete_ai
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

TFLITE_PATH = os.path.join(BASE_DIR, "best_chili_model.tflite")
KERAS_PATH = os.path.join(BASE_DIR, "best_chili_model.keras")
CLASS_FILE = os.path.join(BASE_DIR, "class_names.txt")

with open(CLASS_FILE) as f:
    class_names = [l.strip() for l in f if l.strip()]

print(f"Loaded Classes ({len(class_names)}): {class_names}")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
in_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()

def predict(crop_bgr):
    img = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    interpreter.set_tensor(in_details[0]['index'], np.expand_dims(rgb, 0))
    interpreter.invoke()
    preds = interpreter.get_tensor(out_details[0]['index'])[0]
    idx = int(np.argmax(preds))
    return class_names[idx], preds[idx], preds

print("\n" + "="*60)
print("TESTING ALL 4 CLASSES FROM DATASET (3 DISEASES + HEALTHY)")
print("="*60)

for target_class in class_names:
    samples = glob.glob(f"dataset/{target_class}/*.*")[:5]
    print(f"\n--- Testing Class: {target_class} ({len(samples)} samples) ---")
    correct = 0
    for s in samples:
        img = cv2.imread(s)
        if img is None: continue
        # Center crop or full leaf
        pred_class, conf, all_p = predict(img)
        is_ok = (pred_class == target_class)
        if is_ok: correct += 1
        status = "✅ PASS" if is_ok else "❌ FAIL"
        print(f"  {os.path.basename(s):30s} -> Predicted: {pred_class:20s} ({conf*100:.1f}%) {status}")
    print(f"Result for {target_class}: {correct}/{len(samples)} correct")

print("\n" + "="*60)
print("TESTING NON-LEAF REJECTION (FACES, WALLS, OBJECTS)")
print("="*60)

from complete_ai import detect_leaf

test_non_leaf = [
    ("Human Face & Room Ceiling", "test_rotated.jpg"),
    ("Wall Ledge / Classroom Shelf", "detected_leaves/leaf_1790828616.jpg"),
    ("Earlier Classroom Room Shot", "detected_leaves/leaf_1790752118.jpg")
]

for name, path in test_non_leaf:
    if os.path.exists(path):
        img = cv2.imread(path)
        crop, box, mode = detect_leaf(img)
        print(f"  {name:30s} -> Mode: {mode} (Box: {box})")
        if mode == "NO_LEAF":
            print(f"     ✅ Correctly rejected as non-leaf! Display shows: NO LEAF DETECTED")
        else:
            print(f"     ❌ Incorrectly passed as leaf!")

print("\nAll tests completed.")
