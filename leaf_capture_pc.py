import cv2
import os
import time
import numpy as np


# ============================================================
# CHILI DISEASE AI
# LEAF DETECTION + LEAF CROPPING
# WINDOWS / LAPTOP VERSION
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Minimum area required to consider something a leaf
MIN_LEAF_AREA = 5000

# Padding around detected leaf
PADDING = 20

# Time between automatic captures
CAPTURE_COOLDOWN = 3

# Folder for cropped leaves
SAVE_FOLDER = "cropped_leaves"


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SAVE_FOLDER = os.path.join(
    BASE_DIR,
    SAVE_FOLDER
)

os.makedirs(
    SAVE_FOLDER,
    exist_ok=True
)


# ============================================================
# START CAMERA
# ============================================================

print()
print("=" * 60)
print("CHILI DISEASE AI - LEAF CROPPING SYSTEM")
print("=" * 60)
print()

print("Starting camera...")


camera = cv2.VideoCapture(
    CAMERA_INDEX
)


if not camera.isOpened():

    print()
    print("ERROR: Camera could not be opened.")
    print()
    print("Try changing:")
    print("CAMERA_INDEX = 0")
    print("to:")
    print("CAMERA_INDEX = 1")

    input(
        "\nPress Enter to exit..."
    )

    exit()


# ============================================================
# CAMERA RESOLUTION
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
print("Camera started.")
print()
print("Place one chili leaf in front of the camera.")
print()
print("GREEN BOX = Detected leaf")
print()
print("Q = Quit")
print("S = Save crop manually")
print()


# ============================================================
# TIMER
# ============================================================

last_capture_time = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    success, frame = camera.read()


    if not success:

        print(
            "ERROR: Could not read camera."
        )

        break


    # --------------------------------------------------------
    # CREATE ORIGINAL FRAME COPY
    # --------------------------------------------------------

    original_frame = frame.copy()


    # --------------------------------------------------------
    # HSV CONVERSION
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    # --------------------------------------------------------
    # GREEN RANGE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CREATE MASK
    # --------------------------------------------------------

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )


    # --------------------------------------------------------
    # REMOVE NOISE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FIND CONTOURS
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # --------------------------------------------------------
    # FIND BEST LEAF
    # --------------------------------------------------------

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


    # ========================================================
    # IF LEAF FOUND
    # ========================================================

    if best_contour is not None:

        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        x, y, w, h = cv2.boundingRect(
            best_contour
        )


        # ----------------------------------------------------
        # ADD PADDING
        # ----------------------------------------------------

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


        # ====================================================
        # IMPORTANT:
        # CROP FROM ORIGINAL FRAME
        # NOT FROM DISPLAY FRAME
        # ====================================================

        leaf_crop = original_frame[
            y1:y2,
            x1:x2
        ]


        # ----------------------------------------------------
        # DRAW BOX ONLY FOR DISPLAY
        # ----------------------------------------------------

        display_frame = frame.copy()


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


        # ----------------------------------------------------
        # DISPLAY TEXT
        # ----------------------------------------------------

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

            0.9,

            (
                0,
                255,
                0
            ),

            2
        )


        cv2.putText(

            display_frame,

            "Press S to save",

            (
                20,
                40
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


        # ====================================================
        # AUTOMATIC SAVE
        # ====================================================

        current_time = time.time()


        if (
            current_time
            -
            last_capture_time
            >=
            CAPTURE_COOLDOWN
        ):

            timestamp = time.strftime(
                "%Y%m%d_%H%M%S"
            )


            filename = (

                "leaf_"

                +
                timestamp

                +
                ".jpg"
            )


            filepath = os.path.join(

                SAVE_FOLDER,

                filename
            )


            # ------------------------------------------------
            # SAVE CLEAN CROP
            # ------------------------------------------------

            cv2.imwrite(

                filepath,

                leaf_crop
            )


            print()
            print(
                "========================================"
            )

            print(
                "LEAF CROPPED SUCCESSFULLY"
            )

            print(
                "========================================"
            )

            print(
                "File:",
                filename
            )

            print(
                "Path:",
                filepath
            )

            print(
                "Size:",
                leaf_crop.shape[1],
                "x",
                leaf_crop.shape[0]
            )

            print()


            last_capture_time = (
                current_time
            )


    else:

        # ----------------------------------------------------
        # NO LEAF
        # ----------------------------------------------------

        display_frame = frame.copy()


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


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(

        "Chili Leaf Detection and Cropping",

        display_frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(
        1
    ) & 0xFF


    # --------------------------------------------------------
    # Q = QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break


    # --------------------------------------------------------
    # S = MANUAL SAVE
    # --------------------------------------------------------

    if (
        key == ord("s")
        and
        best_contour is not None
    ):

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )


        filename = (

            "manual_leaf_"

            +
            timestamp

            +
            ".jpg"
        )


        filepath = os.path.join(

            SAVE_FOLDER,

            filename
        )


        cv2.imwrite(

            filepath,

            leaf_crop
        )


        print()
        print(
            "Manual crop saved:"
        )

        print(
            filepath
        )

        print()


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()


print()
print("=" * 60)
print("LEAF CROPPING SYSTEM STOPPED")
print("=" * 60)
print()

print(
    "Cropped images are stored in:"
)

print(
    SAVE_FOLDER
)

print()

input(
    "Press Enter to close..."
)