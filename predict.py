import tensorflow as tf
import numpy as np
from PIL import Image
from tkinter import Tk, filedialog
import json
import os
import shutil

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "chili_disease_model.keras"
IMAGE_SIZE = (224, 224)

# Minimum confidence required
CONFIDENCE_THRESHOLD = 0.60

# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    MODEL_PATH
)

CLASS_NAMES_FILE = os.path.join(
    BASE_DIR,
    "class_names.txt"
)

WEBSITE_FOLDER = os.path.join(
    BASE_DIR,
    "website"
)

RESULT_FILE = os.path.join(
    WEBSITE_FOLDER,
    "result.json"
)

IMAGE_FOLDER = os.path.join(
    WEBSITE_FOLDER,
    "static",
    "images"
)

# Create image folder if it does not exist
os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)

# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading AI model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("AI model loaded successfully.")

# =========================================================
# LOAD CLASS NAMES
# =========================================================

with open(
    CLASS_NAMES_FILE,
    "r",
    encoding="utf-8"
) as f:

    class_names = [
        line.strip()
        for line in f.readlines()
    ]

# =========================================================
# DISEASE INFORMATION
# =========================================================

disease_info = {

    "Bacterial_Spot": {

        "chemical":
            "Copper oxychloride-based bactericide",

        "fertilizer":
            "Balanced NPK fertilizer",

        "treatment":
            "Remove infected leaves and avoid excess moisture."
    },

    "Cercospora_Leaf_Spot": {

        "chemical":
            "Copper-based fungicide",

        "fertilizer":
            "Balanced NPK fertilizer",

        "treatment":
            "Remove infected leaves and maintain good air circulation."
    },

    "Curl_Virus": {

        "chemical":
            "Insecticide for whitefly/vector control",

        "fertilizer":
            "Balanced NPK fertilizer",

        "treatment":
            "Remove severely affected leaves and control insect vectors."
    },

    "Healthy_Leaf": {

        "chemical":
            "No chemical treatment required",

        "fertilizer":
            "Balanced NPK fertilizer",

        "treatment":
            "Leaf appears healthy. Continue normal monitoring."
    }
}

# =========================================================
# SELECT IMAGE
# =========================================================

root = Tk()
root.withdraw()

print("\n========================================")
print("       SELECT CHILI LEAF IMAGE")
print("========================================")

image_path = filedialog.askopenfilename(

    title="Select Chili Leaf Image",

    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png"),
        ("JPEG files", "*.jpg *.jpeg"),
        ("PNG files", "*.png")
    ]
)

root.destroy()

if not image_path:

    print("\nNo image selected.")
    exit()

# =========================================================
# LOAD ORIGINAL IMAGE
# =========================================================

try:

    original_image = Image.open(
        image_path
    ).convert("RGB")

except Exception as e:

    print("\nUnable to open image.")
    print(e)

    exit()

# =========================================================
# COPY ORIGINAL IMAGE TO WEBSITE
# =========================================================

image_filename = os.path.basename(
    image_path
)

website_image_path = os.path.join(
    IMAGE_FOLDER,
    image_filename
)

try:

    shutil.copy2(
        image_path,
        website_image_path
    )

except Exception as e:

    print("\nWarning: Could not copy image to website.")
    print(e)

# =========================================================
# PREPROCESS IMAGE FOR AI
# =========================================================

image = original_image.resize(
    IMAGE_SIZE
)

image_array = np.array(
    image,
    dtype=np.float32
)

image_array = np.expand_dims(
    image_array,
    axis=0
)

# =========================================================
# AI ANALYSIS
# =========================================================

print("\nAnalyzing image...")

predictions = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(
    predictions[0]
)

confidence = predictions[0][
    predicted_index
]

predicted_class = class_names[
    predicted_index
]

# =========================================================
# CONFIDENCE CHECK
# =========================================================

if confidence < CONFIDENCE_THRESHOLD:

    print("\n")
    print("========================================")
    print("        CHILI LEAF AI RESULT")
    print("========================================")

    print("\nImage unclear.")
    print("Please capture a clearer chili leaf image.")

    print("\n========================================")

    # Save unclear result for website
    result = {

        "image": image_filename,

        "disease": "Image Unclear",

        "chemical": "Not recommended",

        "fertilizer": "Not available",

        "treatment":
            "Please capture a clearer chili leaf image."
    }

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )

    print("\nResult saved for website.")

    exit()

# =========================================================
# GET DISEASE INFORMATION
# =========================================================

info = disease_info.get(

    predicted_class,

    {

        "chemical":
            "Not available",

        "fertilizer":
            "Balanced NPK fertilizer",

        "treatment":
            "Please consult an agricultural expert."
    }
)

# =========================================================
# CLEAN TERMINAL OUTPUT
# =========================================================

print("\n")
print("========================================")
print("        CHILI LEAF AI RESULT")
print("========================================")

print("\nThis is:")

print(
    predicted_class.replace(
        "_",
        " "
    )
)

print("\nRecommended Chemical:")

print(
    info["chemical"]
)

print("\nRecommended Fertilizer:")

print(
    info["fertilizer"]
)

print("\nTreatment:")

print(
    info["treatment"]
)

print("\n========================================")

# =========================================================
# SAVE RESULT FOR WEBSITE
# =========================================================

result = {

    "image":
        image_filename,

    "disease":
        predicted_class.replace(
            "_",
            " "
        ),

    "chemical":
        info["chemical"],

    "fertilizer":
        info["fertilizer"],

    "treatment":
        info["treatment"]
}

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=4
    )

print("\nResult saved for website.")
print("Image saved for website.")