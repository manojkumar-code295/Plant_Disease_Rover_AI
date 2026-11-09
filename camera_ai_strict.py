import os
import time
import requests
import cv2
import numpy as np
from picamera2 import Picamera2
from tensorflow.keras.models import load_model

# ============================================================
# PRECISION AGRICULTURE SMART ROVER
# STRICT CHILI LEAF + DISEASE DETECTION
# Raspberry Pi CSI Camera + TensorFlow/Keras + Render
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_chili_model.keras")
CLASS_FILE = os.path.join(BASE_DIR, "class_names.txt")

RENDER_URL = "https://precision-agriculture-smart-rover.onrender.com"
UPDATE_URL = RENDER_URL + "/api/update-result"

IMAGE_SIZE = 224

# Disease-model confidence. This is NOT leaf confidence.
DISEASE_CONFIDENCE_THRESHOLD = 0.60

# Do not upload repeatedly every frame.
UPLOAD_INTERVAL = 2.0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# A valid leaf must be seen consistently for this many frames.
STABLE_FRAMES_REQUIRED = 3

# Minimum contour area relative to whole camera image.
MIN_LEAF_AREA_RATIO = 0.018
MAX_LEAF_AREA_RATIO = 0.55

# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {
    "Bacterial_Spot": {
        "chemical": "Copper oxychloride-based bactericide",
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Remove infected leaves and avoid excess moisture."
    },
    "Cercospora_Leaf_Spot": {
        "chemical": "Copper-based fungicide",
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Remove infected leaves and maintain good air circulation."
    },
    "Curl_Virus": {
        "chemical": "Insecticide for whitefly/vector control",
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Remove severely affected leaves and control insect vectors."
    },
    "Healthy_Leaf": {
        "chemical": "No chemical treatment required",
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Leaf appears healthy. Continue normal monitoring."
    }
}

# ============================================================
# LOAD CLASS NAMES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found: " + MODEL_PATH)

if not os.path.exists(CLASS_FILE):
    raise FileNotFoundError("Class file not found: " + CLASS_FILE)

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

print("=" * 60)
print("MODEL:", MODEL_PATH)
print("CLASS FILE:", CLASS_FILE)
print("CLASS ORDER:")
for i, name in enumerate(class_names):
    print("  ", i, "->", name)
print("=" * 60)

model = load_model(MODEL_PATH, compile=False)

output_count = int(model.output_shape[-1])

if output_count != len(class_names):
    raise RuntimeError(
        "MODEL/CLASS MISMATCH: model has "
        + str(output_count)
        + " outputs but class_names.txt has "
        + str(len(class_names))
        + " classes."
    )

print("Model loaded successfully.")
print("Model output classes:", output_count)

# ============================================================
# CAMERA
# ============================================================

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
        "format": "RGB888"
    }
)

picam2.configure(camera_config)
picam2.start()

time.sleep(2)

# ============================================================
# SEND RESULT TO RENDER
# ============================================================

def send_to_render(image, disease, confidence):
    info = DISEASE_INFO.get(
        disease,
        {
            "chemical": "Not specified",
            "fertilizer": "Balanced NPK fertilizer",
            "treatment": "Continue monitoring the plant."
        }
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    if not ok:
        print("ERROR: Could not encode image.")
        return

    files = {
        "image": (
            "leaf.jpg",
            encoded.tobytes(),
            "image/jpeg"
        )
    }

    data = {
        "disease": disease,
        "chemical": info["chemical"],
        "fertilizer": info["fertilizer"],
        "treatment": info["treatment"],
        "confidence": f"{confidence:.2f}%"
    }

    try:
        response = requests.post(
            UPDATE_URL,
            files=files,
            data=data,
            timeout=15
        )

        if response.ok:
            print()
            print("=" * 60)
            print("RESULT SENT TO RENDER")
            print("Disease   :", disease)
            print("Confidence:", f"{confidence:.2f}%")
            print("=" * 60)
        else:
            print(
                "Render error:",
                response.status_code,
                response.text[:300]
            )

    except Exception as e:
        print("Render connection error:", e)

# ============================================================
# LEAF DETECTION
# ============================================================

def detect_leaf(frame):
    """
    Returns:
        crop, bounding_box, score
    or:
        None, None, 0

    This is deliberately stricter than the previous detector.
    It tries to reject large green background areas and simple
    green objects before the disease model is called.
    """

    if frame is None or frame.size == 0:
        return None, None, 0.0

    height, width = frame.shape[:2]
    frame_area = float(width * height)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Chili leaves are normally green.
    lower_green = np.array([28, 45, 35], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Remove small noise.
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None, 0.0

    best = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        area_ratio = area / frame_area

        if area_ratio < MIN_LEAF_AREA_RATIO:
            continue

        if area_ratio > MAX_LEAF_AREA_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 80 or h < 50:
            continue

        box_area = float(w * h)

        if box_area <= 0:
            continue

        bbox_ratio = min(w, h) / max(w, h)

        # Reject extremely thin strips.
        if bbox_ratio < 0.12:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        # Circularity:
        # 1.0 = circle, smaller = more elongated/irregular.
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

        # Very circular green objects are less leaf-like.
        if circularity > 0.92:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area <= 0:
            continue

        solidity = area / hull_area

        if solidity < 0.55:
            continue

        extent = area / box_area

        if extent < 0.25:
            continue

        if extent > 0.90:
            continue

        # Analyze only the contour pixels.
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(
            contour_mask,
            [contour],
            -1,
            255,
            thickness=-1
        )

        contour_pixels = frame[contour_mask > 0]

        if len(contour_pixels) < 100:
            continue

        contour_hsv = hsv[contour_mask > 0]

        green_ratio = float(np.mean(
            (contour_hsv[:, 0] >= 28) &
            (contour_hsv[:, 0] <= 95) &
            (contour_hsv[:, 1] >= 45)
        ))

        if green_ratio < 0.60:
            continue

        # Reject very uniform large green surfaces.
        # Real leaves usually have some variation from veins,
        # edges, shadows, spots or natural texture.
        gray = cv2.cvtColor(
            contour_pixels.reshape(-1, 1, 3),
            cv2.COLOR_BGR2GRAY
        ).reshape(-1)

        gray_std = float(np.std(gray))

        # A completely flat green object is unlikely to be a leaf.
        if gray_std < 4.0:
            continue

        # Crop with padding.
        padding = 20

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Leaf-likeness score.
        # Higher is better.
        score = (
            min(area_ratio / 0.12, 1.0) * 2.0
            + solidity * 1.5
            + extent * 1.0
            + green_ratio * 2.0
            + min(gray_std / 35.0, 1.0) * 1.0
            - circularity * 1.0
        )

        if score > best_score:
            best_score = score
            best = (
                crop,
                (x1, y1, x2, y2),
                score
            )

    if best is None:
        return None, None, 0.0

    return best

# ============================================================
# DISEASE PREDICTION
# ============================================================

def predict_disease(leaf):
    resized = cv2.resize(
        leaf,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    image = rgb.astype(np.float32)

    image = np.expand_dims(image, axis=0)

    predictions = model.predict(
        image,
        verbose=0
    )[0]

    index = int(np.argmax(predictions))
    confidence = float(predictions[index]) * 100.0

    disease = class_names[index]

    return disease, confidence, predictions

# ============================================================
# OPTIONAL HEALTHY-LEAF VISUAL CHECK
# ============================================================

def looks_clean_green_leaf(leaf):
    """
    Conservative visual check for an obviously clean green leaf.

    This does NOT replace the trained disease model.
    It is only used as a safety check for very clean green leaves.
    """

    hsv = cv2.cvtColor(leaf, cv2.COLOR_BGR2HSV)

    green = (
        (hsv[:, :, 0] >= 28) &
        (hsv[:, :, 0] <= 95) &
        (hsv[:, :, 1] >= 55) &
        (hsv[:, :, 2] >= 35)
    )

    green_ratio = float(np.mean(green))

    # Detect strongly non-green pixels that can represent
    # obvious lesions, dry/brown areas, etc.
    non_green = ~green
    non_green_ratio = float(np.mean(non_green))

    return green_ratio >= 0.72 and non_green_ratio <= 0.28

# ============================================================
# MAIN LOOP
# ============================================================

stable_count = 0
last_leaf_box = None
last_leaf_crop = None
last_upload_time = 0.0

print()
print("=" * 60)
print("CAMERA AI STARTED")
print("Show ONE chili leaf to the camera.")
print("Press Q to quit.")
print("=" * 60)

try:
    while True:

        frame_rgb = picam2.capture_array()

        frame = cv2.cvtColor(
            frame_rgb,
            cv2.COLOR_RGB2BGR
        )

        leaf, box, leaf_score = detect_leaf(frame)

        display = frame.copy()

        if leaf is None:

            stable_count = 0
            last_leaf_box = None
            last_leaf_crop = None

            cv2.putText(
                display,
                "NO LEAF DETECTED",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

        else:

            stable_count += 1

            x1, y1, x2, y2 = box

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                display,
                f"LEAF CANDIDATE {stable_count}/{STABLE_FRAMES_REQUIRED}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Do not classify immediately.
            # Require the leaf to remain detected for several frames.
            if stable_count >= STABLE_FRAMES_REQUIRED:

                disease, confidence, probabilities = predict_disease(leaf)

                # Display all class probabilities for debugging.
                y_text = 75

                for i, name in enumerate(class_names):
                    p = float(probabilities[i]) * 100.0

                    cv2.putText(
                        display,
                        f"{name}: {p:.1f}%",
                        (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    y_text += 25

                cv2.putText(
                    display,
                    f"PREDICTION: {disease} {confidence:.2f}%",
                    (20, y_text + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                now = time.time()

                # Upload only after a valid leaf has been detected.
                if (
                    confidence >= DISEASE_CONFIDENCE_THRESHOLD * 100.0
                    and now - last_upload_time >= UPLOAD_INTERVAL
                ):

                    # Extra safeguard:
                    # If the leaf is visually very clean/green,
                    # and the model is predicting a disease with
                    # unusually high confidence, keep the model
                    # result visible locally but classify it as
                    # Healthy only when the clean-green evidence is strong.
                    #
                    # This is intentionally conservative.
                    final_disease = disease
                    final_confidence = confidence

                    if (
                        disease != "Healthy_Leaf"
                        and confidence >= 90.0
                        and looks_clean_green_leaf(leaf)
                    ):
                        final_disease = "Healthy_Leaf"
                        final_confidence = confidence

                        print()
                        print("VISUAL HEALTH CHECK:")
                        print("The leaf is strongly green/clean.")
                        print("Overriding an obvious high-confidence")
                        print("disease prediction to Healthy_Leaf.")
                        print()

                    send_to_render(
                        leaf,
                        final_disease,
                        final_confidence
                    )

                    last_upload_time = now

                    # Keep this leaf from being repeatedly accepted
                    # immediately.
                    stable_count = 0

        cv2.imshow(
            "Precision Agriculture CSI Camera AI",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print()
    print("Program finished.")
