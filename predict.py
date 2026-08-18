# ============================================================
# CHILI DISEASE AI - PREDICTION PROGRAM
# Precision Agriculture Smart Rover
# ============================================================

import os
import json
import shutil
import mimetypes
import requests
import numpy as np
import tensorflow as tf

from PIL import Image
from tkinter import Tk, filedialog


# ============================================================
# SETTINGS
# ============================================================

# Online Render website
WEBSITE_URL = "https://precision-agriculture-smart-rover.onrender.com"

# AI model
MODEL_PATH = "chili_disease_model.keras"

# Image size expected by the model
IMAGE_SIZE = (224, 224)

# Minimum confidence required
# 80% = strict prediction
CONFIDENCE_THRESHOLD = 0.80


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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


# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    print("\nERROR: AI model not found.")

    print("\nExpected model:")
    print(MODEL_FILE)

    print("\nMake sure this file exists:")
    print("chili_disease_model.keras")

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# CHECK CLASS NAMES
# ============================================================

if not os.path.exists(CLASS_NAMES_FILE):

    print("\nERROR: class_names.txt not found.")

    print("\nExpected file:")
    print(CLASS_NAMES_FILE)

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# LOAD AI MODEL
# ============================================================

print("\n")
print("================================================")
print("        CHILI DISEASE AI MODEL")
print("================================================")

print("\nLoading AI model...")

try:

    model = tf.keras.models.load_model(
        MODEL_FILE
    )

    print("AI model loaded successfully.")

except Exception as e:

    print("\nERROR while loading AI model:")
    print(e)

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# LOAD CLASS NAMES
# ============================================================

try:

    with open(
        CLASS_NAMES_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        class_names = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

except Exception as e:

    print("\nERROR while loading class names:")
    print(e)

    input("\nPress Enter to exit...")
    exit()


print("\nClasses detected by model:")

for i, class_name in enumerate(class_names):

    print(
        f"{i}: {class_name}"
    )


# ============================================================
# DISEASE INFORMATION
# ============================================================

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


# ============================================================
# SELECT IMAGE
# ============================================================

print("\n")
print("================================================")
print("        SELECT CHILI LEAF IMAGE")
print("================================================")

root = Tk()

root.withdraw()

root.attributes(
    "-topmost",
    True
)

image_path = filedialog.askopenfilename(

    title="Select Chili Leaf Image",

    filetypes=[
        (
            "Image files",
            "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG"
        ),
        (
            "JPEG files",
            "*.jpg *.jpeg *.JPG *.JPEG"
        ),
        (
            "PNG files",
            "*.png *.PNG"
        )
    ]
)

root.destroy()


# ============================================================
# CHECK IMAGE SELECTION
# ============================================================

if not image_path:

    print("\nNo image selected.")

    print("\nProgram stopped.")

    input("\nPress Enter to exit...")

    exit()


print("\nSelected image:")
print(image_path)


# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================

try:

    original_image = Image.open(
        image_path
    ).convert("RGB")

    print("\nOriginal image loaded successfully.")

except Exception as e:

    print("\nERROR: Unable to open image.")

    print(e)

    input("\nPress Enter to exit...")

    exit()


# ============================================================
# GET IMAGE FILENAME
# ============================================================

image_filename = os.path.basename(
    image_path
)


# ============================================================
# COPY IMAGE TO LOCAL WEBSITE FOLDER
# ============================================================

website_image_path = os.path.join(
    IMAGE_FOLDER,
    image_filename
)

try:

    shutil.copy2(
        image_path,
        website_image_path
    )

    print("\nImage copied to website folder.")

except Exception as e:

    print("\nWARNING: Could not copy image locally.")

    print(e)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

print("\nPreprocessing image...")

try:

    image = original_image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

except Exception as e:

    print("\nERROR during image preprocessing:")

    print(e)

    input("\nPress Enter to exit...")

    exit()


# ============================================================
# AI PREDICTION
# ============================================================

print("\nAnalyzing image with AI model...")

try:

    predictions = model.predict(
        image_array,
        verbose=0
    )

except Exception as e:

    print("\nERROR during AI prediction:")

    print(e)

    input("\nPress Enter to exit...")

    exit()


# ============================================================
# CLEAN PREDICTION ARRAY
# ============================================================

try:

    probabilities = np.asarray(
        predictions[0],
        dtype=np.float32
    )

except Exception as e:

    print("\nERROR reading model output:")

    print(e)

    input("\nPress Enter to exit...")

    exit()


# ============================================================
# HANDLE MODEL OUTPUT
# ============================================================

# If model returns values that do not look like probabilities,
# convert them using softmax.

probability_sum = float(
    np.sum(probabilities)
)

if (
    np.any(probabilities < 0)
    or
    np.any(probabilities > 1)
    or
    not np.isclose(
        probability_sum,
        1.0,
        atol=0.05
    )
):

    exp_values = np.exp(
        probabilities -
        np.max(probabilities)
    )

    probabilities = (
        exp_values /
        np.sum(exp_values)
    )


# ============================================================
# CHECK CLASS COUNT
# ============================================================

if len(probabilities) != len(class_names):

    print("\nERROR:")
    print(
        "Number of model outputs does not match "
        "number of class names."
    )

    print(
        "\nModel outputs:",
        len(probabilities)
    )

    print(
        "Class names:",
        len(class_names)
    )

    print("\nModel probabilities:")

    print(probabilities)

    input("\nPress Enter to exit...")

    exit()


# ============================================================
# GET PREDICTION
# ============================================================

predicted_index = int(
    np.argmax(
        probabilities
    )
)

confidence = float(
    probabilities[
        predicted_index
    ]
)

predicted_class = class_names[
    predicted_index
]


# ============================================================
# DISPLAY ALL CLASS CONFIDENCES
# ============================================================

print("\n")
print("================================================")
print("             AI CONFIDENCE SCORES")
print("================================================")

for i, class_name in enumerate(class_names):

    print(
        f"{class_name.replace('_', ' '):30s}"
        f": {probabilities[i] * 100:.2f}%"
    )


print("\n================================================")


# ============================================================
# DISPLAY TOP PREDICTION
# ============================================================

print("\nTop prediction:")

print(
    predicted_class.replace(
        "_",
        " "
    )
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)

print(
    f"Required: {CONFIDENCE_THRESHOLD * 100:.0f}%"
)


# ============================================================
# CONFIDENCE CHECK
# ============================================================

if confidence < CONFIDENCE_THRESHOLD:

    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    final_disease = "Image Unclear"

    final_chemical = "Not recommended"

    final_fertilizer = "Not available"

    final_treatment = (
        "Please capture a clearer chili leaf image."
    )


    print("\n")
    print("================================================")
    print("             CHILI LEAF AI RESULT")
    print("================================================")

    print("\nRESULT: IMAGE UNCLEAR")

    print(
        f"\nAI Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Required Confidence: "
        f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
    )

    print(
        "\nThe AI is not confident enough "
        "to identify the disease."
    )

    print(
        "\nPlease capture a clearer chili leaf image."
    )


else:

    # --------------------------------------------------------
    # HIGH CONFIDENCE
    # --------------------------------------------------------

    final_disease = predicted_class.replace(
        "_",
        " "
    )

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

    final_chemical = info[
        "chemical"
    ]

    final_fertilizer = info[
        "fertilizer"
    ]

    final_treatment = info[
        "treatment"
    ]


    print("\n")
    print("================================================")
    print("             CHILI LEAF AI RESULT")
    print("================================================")

    print("\nDisease:")

    print(
        final_disease
    )

    print("\nConfidence:")

    print(
        f"{confidence * 100:.2f}%"
    )

    print("\nRecommended Chemical:")

    print(
        final_chemical
    )

    print("\nRecommended Fertilizer:")

    print(
        final_fertilizer
    )

    print("\nTreatment:")

    print(
        final_treatment
    )


# ============================================================
# CREATE RESULT DATA
# ============================================================

result = {

    "image":
        image_filename,

    "disease":
        final_disease,

    "chemical":
        final_chemical,

    "fertilizer":
        final_fertilizer,

    "treatment":
        final_treatment,

    "confidence":
        round(
            confidence * 100,
            2
        )
}


# ============================================================
# SAVE RESULT LOCALLY
# ============================================================

try:

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

    print("\n")
    print(
        "Local result.json updated successfully."
    )

except Exception as e:

    print("\nWARNING:")
    print(
        "Could not save local result.json."
    )

    print(e)


# ============================================================
# SEND RESULT TO ONLINE WEBSITE
# ============================================================

print("\n")
print("================================================")
print("          UPDATING ONLINE WEBSITE")
print("================================================")

api_url = (
    WEBSITE_URL
    + "/api/update-result"
)


try:

    # Determine correct MIME type
    mime_type, _ = mimetypes.guess_type(
        image_path
    )

    if mime_type is None:

        mime_type = "image/jpeg"


    # Open image for upload
    with open(
        image_path,
        "rb"
    ) as image_file:

        files = {

            "image": (

                image_filename,

                image_file,

                mime_type
            )
        }


        data = {

            "disease":
                final_disease,

            "chemical":
                final_chemical,

            "fertilizer":
                final_fertilizer,

            "treatment":
                final_treatment,

            "confidence":
                f"{confidence * 100:.2f}"
        }


        response = requests.post(

            api_url,

            files=files,

            data=data,

            timeout=60
        )


    # ========================================================
    # SERVER RESPONSE
    # ========================================================

    print("\nServer status code:")

    print(
        response.status_code
    )


    if response.status_code == 200:

        print("\n")
        print(
            "================================================"
        )

        print(
            "       ONLINE WEBSITE UPDATED SUCCESSFULLY!"
        )

        print(
            "================================================"
        )


        # Try to display JSON response
        try:

            server_response = (
                response.json()
            )

            print("\nServer response:")

            print(
                json.dumps(
                    server_response,
                    indent=4
                )
            )

        except Exception:

            print(
                "\nServer response:"
            )

            print(
                response.text
            )


    else:

        print("\n")
        print(
            "================================================"
        )

        print(
            "          ONLINE WEBSITE UPDATE FAILED"
        )

        print(
            "================================================"
        )

        print(
            "\nStatus code:",
            response.status_code
        )

        print(
            "\nResponse:"
        )

        print(
            response.text
        )


except requests.exceptions.Timeout:

    print("\nERROR: Website request timed out.")

    print(
        "The Render service may be waking up."
    )

    print(
        "Please try again."
    )


except requests.exceptions.ConnectionError:

    print("\nERROR: Could not connect to website.")

    print(
        "Check your internet connection."
    )

    print(
        "Website:",
        WEBSITE_URL
    )


except Exception as e:

    print(
        "\nERROR while updating online website:"
    )

    print(e)


# ============================================================
# PROCESS COMPLETE
# ============================================================

print("\n")
print("================================================")
print("                PROCESS COMPLETE")
print("================================================")

print(
    "\nYour prediction has been processed."
)

print(
    "The selected image and AI result "
    "were sent to the online website."
)

print(
    "\nWebsite:"
)

print(
    WEBSITE_URL
)

print("\n")


# ============================================================
# END
# ============================================================

input(
    "Press Enter to close..."
)