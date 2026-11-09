"""
Comprehensive Test Suite for Plant Leaf Disease Detection System.
Tests:
  1. Healthy Leaves
  2. Cercospora Diseased Leaves
  3. Anthracnose Diseased Leaves
  4. Bacterial Spot Diseased Leaves
  5. Human Faces (strict rejection verification)
  6. Animals (strict rejection verification)
  7. Vehicles (strict rejection verification)
  8. Fruits/Objects (strict rejection verification)
  9. Buildings (strict rejection verification)
 10. Low-contrast / Unfamiliar / Noise image (uncertain prediction verification)
"""

import os
import sys
from PIL import Image, ImageFilter
import numpy as np

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import PlantDiseaseDetectionPipeline
from src.leaf_verifier import INVALID_MESSAGE

def run_all_tests():
    pipeline = PlantDiseaseDetectionPipeline()
    samples_dir = os.path.join(BASE_DIR, "test_samples")

    test_cases = [
        # Leaf Categories
        {
            "name": "Test Healthy Leaf",
            "file": os.path.join(samples_dir, "sample_healthy.jpg"),
            "expected_is_leaf": True,
            "expected_prediction": "Healthy",
            "category": "LEAF"
        },
        {
            "name": "Test Cercospora Leaf",
            "file": os.path.join(samples_dir, "sample_cercospora.jpg"),
            "expected_is_leaf": True,
            "expected_prediction": "Cercospora",
            "category": "LEAF"
        },
        {
            "name": "Test Anthracnose Leaf",
            "file": os.path.join(samples_dir, "sample_anthracnose.jpg"),
            "expected_is_leaf": True,
            "expected_prediction": "Anthracnose",
            "category": "LEAF"
        },
        {
            "name": "Test Bacterial Spot Leaf",
            "file": os.path.join(samples_dir, "sample_bacterial_spot.jpg"),
            "expected_is_leaf": True,
            "expected_prediction": "Bacterial_Spot",
            "category": "LEAF"
        },
        # Non-Leaf Rejection Cases
        {
            "name": "Test Human Face (Strict Rejection)",
            "file": os.path.join(samples_dir, "sample_nonleaf_face.jpg"),
            "expected_is_leaf": False,
            "expected_error": INVALID_MESSAGE,
            "category": "NON_LEAF"
        },
        {
            "name": "Test Animal (Baboon) Rejection",
            "file": os.path.join(samples_dir, "sample_nonleaf_animal.jpg"),
            "expected_is_leaf": False,
            "expected_error": INVALID_MESSAGE,
            "category": "NON_LEAF"
        },
        {
            "name": "Test Vehicle (Car) Rejection",
            "file": os.path.join(samples_dir, "sample_nonleaf_car.jpg"),
            "expected_is_leaf": False,
            "expected_error": INVALID_MESSAGE,
            "category": "NON_LEAF"
        },
        {
            "name": "Test Fruit Object Rejection",
            "file": os.path.join(samples_dir, "sample_nonleaf_fruit.jpg"),
            "expected_is_leaf": False,
            "expected_error": INVALID_MESSAGE,
            "category": "NON_LEAF"
        },
        {
            "name": "Test Building Architecture Rejection",
            "file": os.path.join(samples_dir, "sample_nonleaf_building.jpg"),
            "expected_is_leaf": False,
            "expected_error": INVALID_MESSAGE,
            "category": "NON_LEAF"
        }
    ]

    print("\n" + "=" * 70)
    print("      PLANT LEAF DISEASE DETECTION - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    passed_count = 0
    total_tests = len(test_cases) + 1  # including blurred uncertain test

    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i:02d}/{total_tests:02d}] {tc['name']}")
        print(f"       File: {os.path.basename(tc['file'])}")

        if not os.path.exists(tc["file"]):
            print(f"       ❌ FAILED: Sample file not found at {tc['file']}")
            continue

        img = Image.open(tc["file"])
        result = pipeline.process_image(img)

        if tc["category"] == "LEAF":
            is_leaf_ok = result.get("is_leaf") == tc["expected_is_leaf"]
            pred_ok = result.get("prediction") == tc["expected_prediction"]
            conf = result.get("confidence", 0)

            if is_leaf_ok and pred_ok:
                print(f"       ✅ PASSED: Verified as Leaf (Conf: {result.get('leaf_confidence', 0):.1f}%) | "
                      f"Predicted: {result.get('prediction')} ({conf:.1f}%)")
                passed_count += 1
            else:
                print(f"       ❌ FAILED: is_leaf={result.get('is_leaf')}, predicted={result.get('prediction')} "
                      f"(expected {tc['expected_prediction']})")

        elif tc["category"] == "NON_LEAF":
            is_rejected = (result.get("is_leaf") is False)
            err_msg_ok = (result.get("error") == tc["expected_error"] or result.get("message") == tc["expected_error"])

            if is_rejected and err_msg_ok:
                print(f"       ✅ PASSED: Correctly Rejected! Message: \"{result.get('message')}\"")
                print(f"                  Details: {result.get('verifier_details', '')}")
                passed_count += 1
            else:
                print(f"       ❌ FAILED: Image was not properly rejected. result={result}")

    # Test Case 10: Unfamiliar / Ambiguous heavily blurred synthetic image
    print(f"\n[{total_tests:02d}/{total_tests:02d}] Test Low-Contrast / Ambiguous Image (Uncertain / Rejection Check)")
    noise_img = Image.fromarray(np.random.randint(90, 140, (224, 224, 3), dtype=np.uint8)).filter(ImageFilter.GaussianBlur(radius=5))
    noise_res = pipeline.process_image(noise_img)
    if not noise_res.get("is_leaf") or noise_res.get("status") == "uncertain":
        status_msg = noise_res.get("message")
        print(f"       ✅ PASSED: Handled gracefully! Result: \"{status_msg}\"")
        passed_count += 1
    else:
        print(f"       ❌ FAILED: Unexpected prediction on noise image: {noise_res}")

    print("\n" + "=" * 70)
    print(f"  TEST SUMMARY: {passed_count}/{total_tests} Tests Passed ({(passed_count/total_tests)*100:.1f}%)")
    print("=" * 70 + "\n")
    return passed_count == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
