import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "best_chili_model.keras"
CLASS_FILE = "class_names.txt"

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

IMG_SIZE = (224, 224)

# Number of images to test from each class
TEST_IMAGES_PER_CLASS = 10


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_FILE, "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "->", name)


# ============================================================
# CHECK NUMBER OF CLASSES
# ============================================================

if len(class_names) != model.output_shape[-1]:
    print("\nERROR:")
    print("Number of class names does not match model output.")
    print("Class names:", len(class_names))
    print("Model output:", model.output_shape[-1])
    exit()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(img_path):

    img = image.load_img(
        img_path,
        target_size=IMG_SIZE
    )

    img_array = image.img_to_array(img)
    # Model internally runs preprocess_input which expects [0, 255]
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = prediction[predicted_index] * 100

    return predicted_class, confidence, prediction


# ============================================================
# FIND DATASET
# ============================================================

print("\nChecking dataset...")

if not os.path.exists(DATASET_PATH):
    print("\nDATASET PATH NOT FOUND:")
    print(DATASET_PATH)
    print("\nChange DATASET_PATH in this file to your actual dataset folder.")
    exit()


# ============================================================
# TEST EACH CLASS
# ============================================================

print("\n")
print("=" * 70)
print("TESTING MODEL ON DATASET")
print("=" * 70)


total = 0
correct = 0

for actual_class in class_names:

    class_folder = os.path.join(
        DATASET_PATH,
        actual_class
    )

    print("\n")
    print("-" * 70)
    print("ACTUAL CLASS:", actual_class)
    print("-" * 70)

    if not os.path.exists(class_folder):

        print("Folder NOT FOUND:")
        print(class_folder)

        continue

    files = []

    for filename in os.listdir(class_folder):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            files.append(
                os.path.join(
                    class_folder,
                    filename
                )
            )

    files.sort()

    if len(files) == 0:

        print("No images found.")

        continue

    files = files[:TEST_IMAGES_PER_CLASS]

    class_correct = 0

    for img_path in files:

        try:

            predicted_class, confidence, prediction = predict_image(
                img_path
            )

            filename = os.path.basename(img_path)

            result = "WRONG"

            if predicted_class == actual_class:
                result = "CORRECT"
                correct += 1
                class_correct += 1

            total += 1

            print(
                f"{filename[:30]:30s} -> "
                f"{predicted_class:22s} "
                f"{confidence:6.2f}% "
                f"[{result}]"
            )

        except Exception as e:

            print(
                "ERROR:",
                os.path.basename(img_path),
                "->",
                e
            )

    print(
        f"\n{actual_class}: "
        f"{class_correct}/{len(files)} correct"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

if total > 0:

    accuracy = (correct / total) * 100

    print("Images tested :", total)
    print("Correct       :", correct)
    print("Wrong         :", total - correct)
    print("Accuracy      :", f"{accuracy:.2f}%")

else:

    print("No images were tested.")



