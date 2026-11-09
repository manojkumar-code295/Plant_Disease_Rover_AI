import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "best_chili_model.keras"
CLASS_FILE = "class_names.txt"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_FILE, "r") as f:
    class_names = [x.strip() for x in f if x.strip()]

print("\n==============================")
print("MODEL DIAGNOSTIC")
print("==============================")
print("Model input :", model.input_shape)
print("Model output:", model.output_shape)
print("Classes     :", class_names)

# -------------------------------------------------
# Find possible dataset directories
# -------------------------------------------------

possible_dirs = [
    "dataset",
    "data",
    "chili_dataset",
    "chilli_dataset",
    "PlantVillage",
    "Dataset"
]

dataset_dir = None

for d in possible_dirs:
    if os.path.isdir(d):
        dataset_dir = d
        break

if dataset_dir is None:
    print("\nERROR: Dataset directory was not found.")
    print("Current directory:")
    print(os.getcwd())
    print("\nDirectories here:")
    for d in os.listdir("."):
        if os.path.isdir(d):
            print("  ", d)
    print("\nPlease put your dataset directory in this project folder.")
    exit()

print("\nDataset directory:", dataset_dir)

# -------------------------------------------------
# Find images for each class
# -------------------------------------------------

extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

print("\n==============================")
print("TESTING ONE IMAGE FROM EACH CLASS")
print("==============================")

all_correct = True

for class_name in class_names:

    class_dir = os.path.join(dataset_dir, class_name)

    if not os.path.isdir(class_dir):
        print("\nWARNING: Folder not found:")
        print(class_dir)
        all_correct = False
        continue

    images = []

    for ext in extensions:
        images.extend(glob.glob(os.path.join(class_dir, ext)))

    if len(images) == 0:
        print("\nWARNING: No images found in:")
        print(class_dir)
        all_correct = False
        continue

    image_path = images[0]

    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        # MobileNetV2 has an internal preprocess_input layer expecting [0, 255]
        arr = np.array(img).astype(np.float32)
        arr = np.expand_dims(arr, axis=0)

        prediction = model.predict(arr, verbose=0)[0]

        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(prediction[predicted_index]) * 100

        print("\n--------------------------------")
        print("TRUE CLASS :", class_name)
        print("IMAGE      :", image_path)
        print("PREDICTED  :", predicted_class)
        print("CONFIDENCE :", f"{confidence:.2f}%")

        print("\nProbabilities:")

        for i, name in enumerate(class_names):
            print(
                f"  {name:25s}: "
                f"{prediction[i] * 100:.2f}%"
            )

        if predicted_class != class_name:
            print("RESULT     : ❌ WRONG")
            all_correct = False
        else:
            print("RESULT     : ✅ CORRECT")

    except Exception as e:
        print("\nERROR processing:")
        print(image_path)
        print(e)
        all_correct = False

# -------------------------------------------------
# Final result
# -------------------------------------------------

print("\n==============================")
print("FINAL DIAGNOSIS")
print("==============================")

if all_correct:
    print("✅ Model passed the basic class test.")
    print("The problem is probably in the camera/inference pipeline.")
else:
    print("❌ Model failed the basic class test.")
    print("The trained model or its preprocessing/class mapping needs fixing.")

print("==============================")
