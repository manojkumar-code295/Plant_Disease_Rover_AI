import os
import cv2
import numpy as np
import tensorflow as tf
import time

# ============================================================
# CHILI DISEASE AI - LIVE CAMERA TEST
# Raspberry Pi + TensorFlow/Keras
# ============================================================

MODEL_PATH = "best_chili_model.keras"
CLASS_FILE = "class_names.txt"

IMG_SIZE = 224
CAMERA_INDEX = 0

# Predict once every N frames to reduce Raspberry Pi load
PREDICT_EVERY = 8

# Minimum confidence to display a normal prediction
CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# LOAD CLASS NAMES
# ============================================================

if not os.path.exists(CLASS_FILE):
    print("ERROR: class_names.txt not found!")
    print("Current folder:", os.getcwd())
    exit()

with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

print("\nClasses loaded:")

for i, name in enumerate(CLASS_NAMES):
    print(f"{i} -> {name}")


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model not found!")
    print("Expected:", os.path.abspath(MODEL_PATH))
    exit()

print("\nLoading model...")

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
except Exception as e:
    print("ERROR loading model:")
    print(e)
    exit()

print("Model loaded successfully.")

print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

NUM_CLASSES = model.output_shape[-1]

if NUM_CLASSES != len(CLASS_NAMES):
    print("\nWARNING!")
    print("Model classes:", NUM_CLASSES)
    print("Class names  :", len(CLASS_NAMES))
    print("These must be equal.")
    exit()


# ============================================================
# MODEL WARM-UP
# ============================================================

print("\nWarming up model...")

dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

try:
    model.predict(dummy, verbose=0)
except Exception as e:
    print("Model prediction error:")
    print(e)
    exit()

print("Model ready.")


# ============================================================
# OPEN CAMERA
# ============================================================

print("\nStarting camera...")

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("\nERROR: Could not open camera.")
    print("Try changing CAMERA_INDEX from 0 to 1.")
    exit()

# Camera resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera started.")
print("\nControls:")
print("Q = Quit")
print("S = Save current image")
print("R = Force new prediction")
print()


# ============================================================
# VARIABLES
# ============================================================

frame_count = 0

last_prediction = "Starting..."
last_confidence = 0.0
last_probabilities = np.zeros(NUM_CLASSES)

prediction_time = 0

saved_number = 0


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    frame_count += 1

    # --------------------------------------------------------
    # Mirror camera image
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)

    display = frame.copy()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if frame_count % PREDICT_EVERY == 0:

        start_time = time.time()

        try:

            # Convert BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize exactly to model input size
            image = cv2.resize(
                rgb,
                (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_AREA
            )

            # Convert to float
            image = image.astype(np.float32)

            # Add batch dimension
            image = np.expand_dims(image, axis=0)

            # Model prediction
            probabilities = model.predict(
                image,
                verbose=0
            )[0]

            # ------------------------------------------------
            # Get highest probability
            # ------------------------------------------------

            prediction_index = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[prediction_index]
            )

            # Save values
            last_probabilities = probabilities
            last_confidence = confidence

            if confidence >= CONFIDENCE_THRESHOLD:
                last_prediction = CLASS_NAMES[prediction_index]
            else:
                last_prediction = "Uncertain"

            prediction_time = time.time() - start_time

            # ------------------------------------------------
            # Terminal output
            # ------------------------------------------------

            print("\n------------------------------")
            print("LIVE PREDICTION")

            for i, class_name in enumerate(CLASS_NAMES):

                print(
                    f"{class_name:<22} -> "
                    f"{probabilities[i] * 100:.2f}%"
                )

            print("------------------------------")
            print(
                f"FINAL: {last_prediction} | "
                f"CONFIDENCE: {confidence * 100:.2f}%"
            )

            print(
                f"Inference time: "
                f"{prediction_time:.2f} sec"
            )

        except Exception as e:

            print("\nPrediction error:")
            print(e)


    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    # Header
    cv2.rectangle(
        display,
        (0, 0),
        (640, 85),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display,
        "CHILI DISEASE AI",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Prediction
    prediction_text = "Prediction: " + last_prediction

    cv2.putText(
        display,
        prediction_text,
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    # Confidence
    confidence_text = (
        f"Confidence: {last_confidence * 100:.2f}%"
    )

    cv2.putText(
        display,
        confidence_text,
        (390, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )


    # ========================================================
    # DRAW PREDICTION PROBABILITIES
    # ========================================================

    y = 115

    for i, class_name in enumerate(CLASS_NAMES):

        probability = float(
            last_probabilities[i]
        )

        text = (
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )

        cv2.putText(
            display,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        y += 25


    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(
        display,
        "Q: Quit | S: Save | R: Predict",
        (15, display.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Chili Disease AI - Live Camera",
        display
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q") or key == ord("Q"):
        break

    # Save current frame
    elif key == ord("s") or key == ord("S"):

        filename = (
            f"live_test_{saved_number}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        print(
            f"\nImage saved: {filename}"
        )

        saved_number += 1

    # Force prediction
    elif key == ord("r") or key == ord("R"):

        frame_count = PREDICT_EVERY - 1

        print("\nNew prediction requested...")


# ============================================================
# CLEAN SHUTDOWN
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("\nCamera stopped.")
print("CHILI DISEASE AI finished.")
