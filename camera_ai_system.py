# ============================================================
# PRECISION AGRICULTURE SMART ROVER
# CAMERA AI SYSTEM
# CAMERA -> AI MODEL -> FLASK WEBSITE -> APP
# ============================================================

import cv2
import numpy as np
import tensorflow as tf
import requests
import os
import time
import json
from datetime import datetime


# ============================================================
# SETTINGS
# ============================================================

WEBSITE_URL = (
    "https://precision-agriculture-smart-rover.onrender.com"
)

API_URL = (
    WEBSITE_URL + "/api/update-result"
)


# ============================================================
# AI MODEL SETTINGS
# ============================================================

MODEL_PATH = "chili_disease_model.keras"

CLASS_NAMES_FILE = "class_names.txt"

IMAGE_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# LEAF DETECTION SETTINGS
# ============================================================

MIN_LEAF_AREA = 5000

PADDING = 20


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280

CAMERA_HEIGHT = 720


# ============================================================
# WEBSITE UPDATE SETTINGS
# ============================================================

# Send one result every 2 seconds after successful upload.
SEND_COOLDOWN = 2.0

# Timeout for Render request.
REQUEST_TIMEOUT = 30


# ============================================================
# PROJECT PATH
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


# ============================================================
# CAMERA IMAGE STORAGE
# ============================================================

CAPTURE_FOLDER = os.path.join(
    BASE_DIR,
    "camera_predictions"
)


os.makedirs(
    CAPTURE_FOLDER,
    exist_ok=True
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
# LOAD AI MODEL
# ============================================================

print()
print("=" * 70)
print("       PRECISION AGRICULTURE SMART ROVER")
print("              CAMERA AI SYSTEM")
print("=" * 70)
print()

print("Loading AI model...")

try:

    model = tf.keras.models.load_model(
        MODEL_FILE
    )

except Exception as e:

    print()
    print("ERROR: Could not load AI model.")
    print()
    print("Model file:")
    print(MODEL_FILE)
    print()
    print("Error:")
    print(e)

    input("\nPress Enter to exit...")

    raise SystemExit


print("AI model loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

print()
print("Loading class names...")

try:

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

except Exception as e:

    print()
    print("ERROR: Could not load class_names.txt")
    print(e)

    input("\nPress Enter to exit...")

    raise SystemExit


print()
print("Loaded classes:")

for index, class_name in enumerate(
    class_names
):

    print(
        f"{index} -> {class_name}"
    )


# ============================================================
# OPEN CAMERA
# ============================================================

print()
print("Starting camera...")


camera = cv2.VideoCapture(
    CAMERA_INDEX
)


if not camera.isOpened():

    print()
    print("ERROR: Camera could not be opened.")
    print()
    print("Try:")
    print("CAMERA_INDEX = 0")
    print("or")
    print("CAMERA_INDEX = 1")

    input("\nPress Enter to exit...")

    raise SystemExit


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)


print()
print("Camera started successfully.")
print()
print("----------------------------------------------")
print("Place ONE chili leaf in front of the camera.")
print("----------------------------------------------")
print()
print("Q = Quit")
print()


# ============================================================
# REQUEST SESSION
# ============================================================

session = requests.Session()


# ============================================================
# TIMER
# ============================================================

last_sent_time = 0.0


# ============================================================
# SERVER STATUS
# ============================================================

server_online = False


# ============================================================
# FUNCTION: SEND RESULT TO WEBSITE
# ============================================================

def send_to_website(
    image_path,
    disease,
    chemical,
    fertilizer,
    treatment,
    confidence
):

    global server_online

    print()
    print("=" * 70)
    print("             SENDING AI RESULT")
    print("=" * 70)

    try:

        with open(
            image_path,
            "rb"
        ) as image_file:

            files = {

                "image": (

                    os.path.basename(
                        image_path
                    ),

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
                        round(
                            confidence,
                            2
                        )
                    )
            }


            response = session.post(

                API_URL,

                files=files,

                data=data,

                timeout=REQUEST_TIMEOUT
            )


        print()
        print(
            "Server status:",
            response.status_code
        )


        # ====================================================
        # SUCCESS
        # ====================================================

        if response.status_code == 200:

            server_online = True

            print()
            print("=" * 70)
            print("       WEBSITE / APP UPDATED SUCCESSFULLY")
            print("=" * 70)

            try:

                server_data = response.json()

                print()

                print(
                    json.dumps(
                        server_data,
                        indent=4
                    )
                )

            except Exception:

                print(
                    response.text
                )

            return True


        # ====================================================
        # SERVER ERROR
        # ====================================================

        else:

            server_online = False

            print()
            print("WEBSITE UPDATE FAILED")

            print(
                response.text
            )

            return False


    except requests.exceptions.Timeout:

        server_online = False

        print()
        print("SERVER REQUEST TIMEOUT")

        print(
            "Render may be waking up."
        )

        return False


    except requests.exceptions.ConnectionError:

        server_online = False

        print()
        print("COULD NOT CONNECT TO WEBSITE")

        print(
            "Check internet connection."
        )

        return False


    except Exception as e:

        server_online = False

        print()
        print("WEBSITE ERROR:")
        print(e)

        return False


# ============================================================
# FUNCTION: GET DISEASE INFORMATION
# ============================================================

def get_disease_information(
    predicted_class,
    confidence
):

    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        return {

            "disease":
                "Image Unclear",

            "chemical":
                "Not recommended",

            "fertilizer":
                "Not available",

            "treatment":
                "Please capture a clearer leaf image."
        }


    # --------------------------------------------------------
    # NORMAL RESULT
    # --------------------------------------------------------

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


    return {

        "disease":
            disease,

        "chemical":
            info["chemical"],

        "fertilizer":
            info["fertilizer"],

        "treatment":
            info["treatment"]
    }


# ============================================================
# FUNCTION: PREDICT LEAF
# ============================================================

def predict_leaf(
    leaf_crop
):

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    leaf_rgb = cv2.cvtColor(

        leaf_crop,

        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    resized = cv2.resize(

        leaf_rgb,

        IMAGE_SIZE
    )


    # --------------------------------------------------------
    # FLOAT32
    # --------------------------------------------------------

    image_array = np.array(

        resized,

        dtype=np.float32
    )


    # --------------------------------------------------------
    # BATCH
    # --------------------------------------------------------

    image_array = np.expand_dims(

        image_array,

        axis=0
    )


    # --------------------------------------------------------
    # AI MODEL
    # --------------------------------------------------------

    predictions = model.predict(

        image_array,

        verbose=0
    )


    # --------------------------------------------------------
    # PREDICTED CLASS
    # --------------------------------------------------------

    predicted_index = int(

        np.argmax(
            predictions[0]
        )
    )


    # --------------------------------------------------------
    # CHECK CLASS INDEX
    # --------------------------------------------------------

    if predicted_index >= len(
        class_names
    ):

        raise ValueError(
            "Model class index does not match class_names.txt"
        )


    predicted_class = class_names[
        predicted_index
    ]


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = float(

        predictions[0][
            predicted_index
        ]
    )


    confidence_percent = (
        confidence * 100
    )


    return (
        predicted_class,
        confidence,
        confidence_percent
    )


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

try:

    while True:

        # ====================================================
        # READ CAMERA
        # ====================================================

        success, frame = camera.read()


        if not success:

            print(
                "Could not read camera frame."
            )

            time.sleep(0.1)

            continue


        # ====================================================
        # ORIGINAL FRAME
        # ====================================================

        original_frame = frame.copy()


        # ====================================================
        # HSV
        # ====================================================

        hsv = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2HSV
        )


        # ====================================================
        # GREEN MASK
        # ====================================================

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


        # ====================================================
        # REMOVE NOISE
        # ====================================================

        kernel = np.ones(

            (
                5,
                5
            ),

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


        # ====================================================
        # FIND CONTOURS
        # ====================================================

        contours, _ = cv2.findContours(

            mask,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE
        )


        # ====================================================
        # FIND LARGEST LEAF
        # ====================================================

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


        # ====================================================
        # DISPLAY FRAME
        # ====================================================

        display_frame = frame.copy()


        # ====================================================
        # LEAF FOUND
        # ====================================================

        if best_contour is not None:

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x, y, w, h = cv2.boundingRect(

                best_contour
            )


            # ------------------------------------------------
            # PADDING
            # ------------------------------------------------

            x1 = max(

                0,

                x - PADDING
            )


            y1 = max(

                0,

                y - PADDING
            )


            x2 = min(

                frame.shape[1],

                x + w + PADDING
            )


            y2 = min(

                frame.shape[0],

                y + h + PADDING
            )


            # ------------------------------------------------
            # CROP
            # ------------------------------------------------

            leaf_crop = original_frame[

                y1:y2,

                x1:x2
            ]


            if leaf_crop.size == 0:

                continue


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(

                display_frame,

                (
                    x1,
                    y1
                ),

                (
                    x2,
                    y2
                ),

                (
                    0,
                    255,
                    0
                ),

                3
            )


            # ------------------------------------------------
            # LEAF DETECTED
            # ------------------------------------------------

            cv2.putText(

                display_frame,

                "LEAF DETECTED",

                (
                    x1,
                    max(
                        30,
                        y1 - 10
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (
                    0,
                    255,
                    0
                ),

                2
            )


            # =================================================
            # AI PREDICTION
            # =================================================

            try:

                (
                    predicted_class,
                    confidence,
                    confidence_percent
                ) = predict_leaf(

                    leaf_crop
                )


                # ------------------------------------------------
                # DISEASE INFORMATION
                # ------------------------------------------------

                result = get_disease_information(

                    predicted_class,

                    confidence
                )


                disease = result[
                    "disease"
                ]

                chemical = result[
                    "chemical"
                ]

                fertilizer = result[
                    "fertilizer"
                ]

                treatment = result[
                    "treatment"
                ]


                # =================================================
                # DISPLAY RESULT
                # =================================================

                if confidence >= CONFIDENCE_THRESHOLD:

                    result_color = (

                        0,
                        255,
                        0
                    )

                else:

                    result_color = (

                        0,
                        0,
                        255
                    )


                cv2.putText(

                    display_frame,

                    "AI: " + disease,

                    (
                        20,
                        80
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    result_color,

                    2
                )


                cv2.putText(

                    display_frame,

                    "Confidence: "
                    +
                    f"{confidence_percent:.2f}%",

                    (
                        20,
                        115
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (
                        255,
                        255,
                        0
                    ),

                    2
                )


                # =================================================
                # SERVER STATUS ON CAMERA
                # =================================================

                if server_online:

                    cv2.putText(

                        display_frame,

                        "SERVER: ONLINE",

                        (
                            20,
                            150
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.65,

                        (
                            0,
                            255,
                            0
                        ),

                        2
                    )

                else:

                    cv2.putText(

                        display_frame,

                        "SERVER: WAITING",

                        (
                            20,
                            150
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.65,

                        (
                            0,
                            0,
                            255
                        ),

                        2
                    )


                # =================================================
                # AUTOMATIC WEBSITE UPDATE
                # =================================================

                current_time = time.time()


                if (

                    current_time
                    -
                    last_sent_time

                    >=

                    SEND_COOLDOWN

                ):

                    # ---------------------------------------------
                    # UNIQUE FILE NAME
                    # ---------------------------------------------

                    timestamp = datetime.now().strftime(

                        "%Y%m%d_%H%M%S"
                    )


                    unique_id = str(

                        int(
                            time.time() * 1000
                        )
                    )


                    filename = (

                        "camera_leaf_"

                        +
                        timestamp

                        +
                        "_"

                        +
                        unique_id

                        +
                        ".jpg"
                    )


                    image_path = os.path.join(

                        CAPTURE_FOLDER,

                        filename
                    )


                    # ---------------------------------------------
                    # SAVE IMAGE
                    # ---------------------------------------------

                    saved = cv2.imwrite(

                        image_path,

                        leaf_crop
                    )


                    if saved:

                        print()
                        print("=" * 70)
                        print("             NEW CAMERA AI RESULT")
                        print("=" * 70)
                        print()
                        print(
                            "Disease:",
                            disease
                        )
                        print(
                            "Confidence:",
                            f"{confidence_percent:.2f}%"
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
                        print(
                            "Image:",
                            filename
                        )


                        # -----------------------------------------
                        # SEND RESULT
                        # -----------------------------------------

                        upload_success = send_to_website(

                            image_path,

                            disease,

                            chemical,

                            fertilizer,

                            treatment,

                            confidence_percent
                        )


                        # -----------------------------------------
                        # UPDATE TIMER ONLY AFTER SUCCESS
                        # -----------------------------------------

                        if upload_success:

                            last_sent_time = (
                                current_time
                            )

                            print()
                            print(
                                "LIVE CAMERA RESULT SENT TO WEBSITE."
                            )

                        else:

                            print()
                            print(
                                "Result was not uploaded."
                            )


                    else:

                        print(
                            "ERROR: Could not save camera image."
                        )


            except Exception as e:

                print()
                print(
                    "AI prediction error:"
                )

                print(
                    str(e)
                )


                cv2.putText(

                    display_frame,

                    "AI ERROR",

                    (
                        20,
                        80
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (
                        0,
                        0,
                        255
                    ),

                    2
                )


        # ====================================================
        # NO LEAF FOUND
        # ====================================================

        else:

            cv2.putText(

                display_frame,

                "SEARCHING FOR LEAF...",

                (
                    20,
                    40
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (
                    0,
                    0,
                    255
                ),

                2
            )


            cv2.putText(

                display_frame,

                "Place one chili leaf in front of camera",

                (
                    20,
                    75
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (
                    255,
                    255,
                    255
                ),

                2
            )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(

            "Precision Agriculture - Camera AI",

            display_frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(
            1
        ) & 0xFF


        # ====================================================
        # QUIT
        # ====================================================

        if key == ord("q"):

            print()
            print(
                "Stopping camera..."
            )

            break


finally:

    # ========================================================
    # RELEASE CAMERA
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()


    print()
    print("=" * 70)
    print("       CAMERA AI SYSTEM STOPPED")
    print("=" * 70)
    