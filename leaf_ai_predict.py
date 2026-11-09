import tensorflow as tf
import numpy as np
import cv2
import os
import json
import requests
from PIL import Image
from tkinter import Tk, filedialog


# ============================================================
# SETTINGS
# ============================================================

WEBSITE_URL = (
    "https://precision-agriculture-smart-rover.onrender.com"
)

MODEL_PATH = "chili_disease_model.keras"

CLASS_NAMES_FILE = "class_names.txt"

IMAGE_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.60

MIN_LEAF_AREA = 5000

PADDING = 20

CROP_FOLDER = "cropped_leaves"


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

CLASS_FILE = os.path.join(
    BASE_DIR,
    CLASS_NAMES_FILE
)

CROP_FOLDER_PATH = os.path.join(
    BASE_DIR,
    CROP_FOLDER
)

os.makedirs(
    CROP_FOLDER_PATH,
    exist_ok=True
)


# ============================================================
# LOAD AI MODEL
# ============================================================

print()
print("=" * 60)
print("       CHILI DISEASE AI SYSTEM")
print("=" * 60)
print()

print("Loading AI model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("AI model loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_FILE,
    "r",
    encoding="utf-8"
) as file:

    class_names = [
        line.strip()
        for line in file.readlines()
        if line.strip()
    ]


print()
print("Classes:")

for index, name in enumerate(
    class_names
):

    print(
        index,
        "->",
        name
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

root = Tk()

root.withdraw()

print()
print("=" * 60)
print("             SELECT LEAF IMAGE")
print("=" * 60)

image_path = filedialog.askopenfilename(

    title="Select Chili Leaf Image",

    filetypes=[

        (
            "Image files",
            "*.jpg *.jpeg *.png"
        ),

        (
            "JPEG files",
            "*.jpg *.jpeg"
        ),

        (
            "PNG files",
            "*.png"
        )
    ]
)

root.destroy()


if not image_path:

    print()
    print("No image selected.")

    exit()


print()
print(
    "Selected image:"
)

print(
    image_path
)


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    original_pil = Image.open(
        image_path
    ).convert("RGB")

except Exception as e:

    print()
    print(
        "ERROR: Unable to open image."
    )

    print(e)

    exit()


# ============================================================
# CONVERT PIL → OPENCV
# ============================================================

original_rgb = np.array(
    original_pil
)

original_bgr = cv2.cvtColor(
    original_rgb,
    cv2.COLOR_RGB2BGR
)


# ============================================================
# LEAF DETECTION
# ============================================================

print()
print(
    "Detecting leaf..."
)


hsv = cv2.cvtColor(
    original_bgr,
    cv2.COLOR_BGR2HSV
)


# ============================================================
# GREEN MASK
# ============================================================

lower_green = np.array([
    25,
    35,
    30
])

upper_green = np.array([
    95,
    255,
    255
])


mask = cv2.inRange(
    hsv,
    lower_green,
    upper_green
)


# ============================================================
# REMOVE NOISE
# ============================================================

kernel = np.ones(
    (5, 5),
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


# ============================================================
# FIND CONTOURS
# ============================================================

contours, _ = cv2.findContours(

    mask,

    cv2.RETR_EXTERNAL,

    cv2.CHAIN_APPROX_SIMPLE
)


# ============================================================
# FIND LARGEST LEAF REGION
# ============================================================

best_contour = None

best_area = 0


for contour in contours:

    area = cv2.contourArea(
        contour
    )


    if area < MIN_LEAF_AREA:

        continue


    if area > best_area:

        best_area = area

        best_contour = contour


# ============================================================
# CREATE CROP
# ============================================================

if best_contour is not None:

    x, y, w, h = cv2.boundingRect(
        best_contour
    )


    # --------------------------------------------
    # PADDING
    # --------------------------------------------

    x1 = max(
        0,
        x - PADDING
    )

    y1 = max(
        0,
        y - PADDING
    )

    x2 = min(
        original_bgr.shape[1],
        x + w + PADDING
    )

    y2 = min(
        original_bgr.shape[0],
        y + h + PADDING
    )


    leaf_crop = original_bgr[
        y1:y2,
        x1:x2
    ]


    print(
        "Leaf detected successfully."
    )

else:

    print()
    print(
        "WARNING: Leaf could not be detected."
    )

    print(
        "Using the complete image instead."
    )

    leaf_crop = original_bgr


# ============================================================
# SAVE CROPPED LEAF
# ============================================================

original_filename = os.path.basename(
    image_path
)

name_without_extension = os.path.splitext(
    original_filename
)[0]


crop_filename = (
    name_without_extension
    +
    "_leaf_crop.jpg"
)


crop_path = os.path.join(
    CROP_FOLDER_PATH,
    crop_filename
)


cv2.imwrite(
    crop_path,
    leaf_crop
)


print()
print(
    "Cropped leaf saved:"
)

print(
    crop_path
)


# ============================================================
# PREPROCESS CROPPED LEAF
# ============================================================

print()
print(
    "Sending cropped leaf to AI model..."
)


leaf_rgb = cv2.cvtColor(
    leaf_crop,
    cv2.COLOR_BGR2RGB
)


leaf_pil = Image.fromarray(
    leaf_rgb
)


leaf_pil = leaf_pil.resize(
    IMAGE_SIZE
)


image_array = np.array(
    leaf_pil,
    dtype=np.float32
)


image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# AI PREDICTION
# ============================================================

predictions = model.predict(
    image_array,
    verbose=0
)


predicted_index = int(
    np.argmax(
        predictions[0]
    )
)


confidence = float(
    predictions[0][
        predicted_index
    ]
)


predicted_class = class_names[
    predicted_index
]


confidence_percent = (
    confidence * 100
)


# ============================================================
# CONFIDENCE CHECK
# ============================================================

print()
print("=" * 60)
print("              AI RESULT")
print("=" * 60)

print()

print(
    "Prediction:",
    predicted_class
)

print(
    "Confidence:",
    f"{confidence_percent:.2f}%"
)


# ============================================================
# LOW CONFIDENCE
# ============================================================

if confidence < CONFIDENCE_THRESHOLD:

    disease = "Image Unclear"

    chemical = "Not recommended"

    fertilizer = "Not available"

    treatment = (
        "Please capture a clearer leaf image."
    )

else:

    disease = predicted_class.replace(
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


    chemical = info[
        "chemical"
    ]

    fertilizer = info[
        "fertilizer"
    ]

    treatment = info[
        "treatment"
    ]


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print()

print(
    "Disease:",
    disease
)

print(
    "Chemical:",
    chemical
)

print(
    "Fertilizer:",
    fertilizer
)

print(
    "Treatment:",
    treatment
)

print()

print("=" * 60)


# ============================================================
# UPLOAD TO WEBSITE
# ============================================================

print()
print(
    "Updating online website..."
)

print(
    WEBSITE_URL
)


api_url = (
    WEBSITE_URL
    +
    "/api/update-result"
)


try:

    with open(
        crop_path,
        "rb"
    ) as image_file:

        files = {

            "image": (

                crop_filename,

                image_file,

                "image/jpeg"
            )
        }


        data = {

            "disease":
                disease,

            "chemical":
                chemical,

            "fertilizer":
                fertilizer,

            "treatment":
                treatment,

            "confidence":
                str(
                    confidence_percent
                )
        }


        response = requests.post(

            api_url,

            files=files,

            data=data,

            timeout=60
        )


    print()

    print(
        "Server status code:",
        response.status_code
    )


    if response.status_code == 200:

        print()
        print("=" * 60)
        print(
            "   WEBSITE UPDATED SUCCESSFULLY"
        )
        print("=" * 60)

        try:

            server_response = (
                response.json()
            )

            print()

            print(
                json.dumps(
                    server_response,
                    indent=4
                )
            )

        except Exception:

            print(
                response.text
            )


    else:

        print()
        print("=" * 60)
        print(
            "       WEBSITE UPDATE FAILED"
        )
        print("=" * 60)

        print()

        print(
            response.text
        )


except requests.exceptions.Timeout:

    print()
    print(
        "ERROR: Website request timed out."
    )

    print(
        "The Render server may be waking up."
    )


except requests.exceptions.ConnectionError:

    print()
    print(
        "ERROR: Could not connect to website."
    )

    print(
        "Check your internet connection."
    )


except Exception as e:

    print()
    print(
        "ERROR while updating website:"
    )

    print(
        str(e)
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("             PROCESS COMPLETE")
print("=" * 60)

print()

print(
    "Cropped image:"
)

print(
    crop_path
)

print()

input(
    "Press Enter to close..."
)