import os
import time
import requests
import cv2
import numpy as np
from picamera2 import Picamera2
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_chili_model.keras")
CLASS_FILE = os.path.join(BASE_DIR, "class_names.txt")

RENDER_URL = "https://precision-agriculture-smart-rover.onrender.com"
UPDATE_URL = RENDER_URL + "/api/update-result"

IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60
UPLOAD_INTERVAL = 1.0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

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

if not os.path.exists(CLASS_FILE):
    print("ERROR: class_names.txt not found")
    raise SystemExit

with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

print("--------------------------------------------")
print("CLASS NAMES:")
for i, name in enumerate(CLASS_NAMES):
    print(i, "->", name)
print("--------------------------------------------")

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found:")
    print(MODEL_PATH)
    raise SystemExit

print("Loading AI model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("AI MODEL LOADED")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

MODEL_CLASSES = model.output_shape[-1]

if MODEL_CLASSES != len(CLASS_NAMES):
    print()
    print("ERROR:")
    print("Model outputs", MODEL_CLASSES, "classes")
    print("class_names.txt contains", len(CLASS_NAMES), "classes")
    print()
    print("Please make sure the correct class_names.txt is being used.")
    raise SystemExit

print()
print("Starting Raspberry Pi camera...")

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

print("CAMERA STARTED")
print("Press Q to stop.")
print("--------------------------------------------")

last_upload_time = 0
running = True

def send_to_render(image, disease, confidence):
    try:
        info = DISEASE_INFO.get(
            disease,
            {
                "chemical": "Not available",
                "fertilizer": "Balanced NPK fertilizer",
                "treatment": "Continue monitoring the plant."
            }
        )

        success, encoded_image = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

        if not success:
            print("ERROR: Could not encode image")
            return False

        image_bytes = encoded_image.tobytes()

        files = {
            "image": (
                "leaf.jpg",
                image_bytes,
                "image/jpeg"
            )
        }

        data = {
            "disease": disease,
            "chemical": info["chemical"],
            "fertilizer": info["fertilizer"],
            "treatment": info["treatment"],
            "confidence": f"{confidence:.2f}"
        }

        response = requests.post(
            UPDATE_URL,
            files=files,
            data=data,
            timeout=15
        )

        if response.status_code == 200:
            print()
            print("--------------------------------------------")
            print("RESULT SENT TO RENDER")
            print("Disease   :", disease)
            print("Confidence:", f"{confidence:.2f}%")
            print("--------------------------------------------")
            return True

        print()
        print("RENDER ERROR")
        print("HTTP:", response.status_code)
        print(response.text[:300])
        return False

    except Exception as e:
        print()
        print("RENDER CONNECTION ERROR:")
        print(e)
        return False

def detect_leaf(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array(
        [25, 55, 35],
        dtype=np.uint8
    )

    upper_green = np.array(
        [100, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    kernel = np.ones(
        (7, 7),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    frame_area = frame.shape[0] * frame.shape[1]

    best_candidate = None
    best_score = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        area_ratio = area / float(frame_area)

        if area_ratio < 0.015:
            continue

        if area_ratio > 0.65:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 60 or h < 40:
            continue

        box_area = w * h

        if box_area <= 0:
            continue

        box_ratio = box_area / float(frame_area)

        if box_ratio > 0.80:
            continue

        aspect_ratio = w / float(h)

        if aspect_ratio > 5.0 or aspect_ratio < 0.20:
            continue

        hull = cv2.convexHull(contour)

        hull_area = cv2.contourArea(hull)

        if hull_area <= 0:
            continue

        solidity = area / float(hull_area)

        if solidity < 0.55:
            continue

        extent = area / float(box_area)

        if extent < 0.20:
            continue

        if extent > 0.95:
            continue

        contour_mask = np.zeros(
            mask.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            contour_mask,
            [contour],
            -1,
            255,
            -1
        )

        green_pixels = cv2.countNonZero(
            cv2.bitwise_and(
                mask,
                contour_mask
            )
        )

        contour_pixels = cv2.countNonZero(
            contour_mask
        )

        if contour_pixels == 0:
            continue

        green_ratio = green_pixels / float(contour_pixels)

        if green_ratio < 0.35:
            continue

        score = (
            area_ratio * 2.0
            + solidity * 0.5
            + extent * 0.5
            + green_ratio * 2.0
        )

        if score > best_score:
            best_score = score
            best_candidate = (
                x,
                y,
                w,
                h,
                contour
            )

    if best_candidate is None:
        return None, mask

    x, y, w, h, contour = best_candidate

    padding = 15

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, mask

    return (
        crop,
        (x1, y1, x2, y2)
    ), mask

def predict_disease(leaf):
    try:
        image = cv2.resize(
            leaf,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(
            np.float32
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        predictions = model.predict(
            image,
            verbose=0
        )[0]

        index = int(
            np.argmax(predictions)
        )

        confidence = float(
            predictions[index]
        ) * 100.0

        disease = CLASS_NAMES[index]

        return disease, confidence

    except Exception as e:
        print("AI prediction error:")
        print(e)
        return None, 0.0

try:
    while running:
        frame_rgb = picam2.capture_array()

        if frame_rgb is None:
            print("Camera returned empty frame")
            time.sleep(0.2)
            continue

        frame = cv2.cvtColor(
            frame_rgb,
            cv2.COLOR_RGB2BGR
        )

        result, mask = detect_leaf(frame)

        display_frame = frame.copy()

        if result is None:
            cv2.putText(
                display_frame,
                "NO LEAF DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        else:
            leaf, box = result

            x1, y1, x2, y2 = box

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                display_frame,
                "LEAF DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            disease, confidence = predict_disease(leaf)

            if disease is not None:
                text = (
                    disease
                    + " : "
                    + f"{confidence:.2f}%"
                )

                cv2.putText(
                    display_frame,
                    text,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 0),
                    2
                )

                if confidence >= (
                    CONFIDENCE_THRESHOLD * 100
                ):
                    current_time = time.time()

                    if (
                        current_time
                        - last_upload_time
                        >= UPLOAD_INTERVAL
                    ):
                        sent = send_to_render(
                            leaf,
                            disease,
                            confidence
                        )

                        if sent:
                            last_upload_time = current_time

                else:
                    cv2.putText(
                        display_frame,
                        "LOW CONFIDENCE",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 165, 255),
                        2
                    )

        cv2.imshow(
            "Precision Agriculture CSI Camera AI",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print()
            print("Q pressed.")
            print("Stopping camera...")
            running = False

except KeyboardInterrupt:
    print()
    print("Stopped by user.")

except Exception as e:
    print()
    print("MAIN ERROR:")
    print(e)

finally:
    print()
    print("Stopping camera...")

    try:
        picam2.stop()
    except Exception:
        pass

    cv2.destroyAllWindows()

    print("Camera stopped.")
    print("Program finished.")
