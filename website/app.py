# ============================================================
# PRECISION AGRICULTURE SMART ROVER
# COMPLETE FLASK WEBSITE BACKEND
# ============================================================

import os
import json
import uuid
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.config["JSON_SORT_KEYS"] = False


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TEMPLATE_FOLDER = os.path.join(
    BASE_DIR,
    "templates"
)

STATIC_FOLDER = os.path.join(
    BASE_DIR,
    "static"
)

IMAGE_FOLDER = os.path.join(
    STATIC_FOLDER,
    "images"
)

RESULT_FILE = os.path.join(
    BASE_DIR,
    "result.json"
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "history.json"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app.template_folder = TEMPLATE_FOLDER
app.static_folder = STATIC_FOLDER


# ============================================================
# DEFAULT RESULT
# ============================================================

DEFAULT_RESULT = {

    "image": "",

    "disease": "No prediction yet",

    "chemical": "Not available",

    "fertilizer": "Not available",

    "treatment": "No prediction has been processed yet.",

    "confidence": 0.0,

    "timestamp": ""
}


# ============================================================
# DEFAULT ROVER STATE
# ============================================================

rover_state = {

    "connected": False,

    "direction": "Stopped",

    "speed": 0,

    "pump": False,

    "mode": "Manual"
}


# ============================================================
# DEFAULT SETTINGS
# ============================================================

rover_settings = {

    "auto_mode": False,

    "spray_enabled": True,

    "detection_enabled": True,

    "confidence_threshold": 80,

    "speed": 50
}


# ============================================================
# JSON HELPERS
# ============================================================

def load_json_file(
    file_path,
    default_value
):

    try:

        if not os.path.exists(
            file_path
        ):

            return default_value

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            "JSON READ ERROR:",
            file_path,
            e
        )

        return default_value


def save_json_file(
    file_path,
    data
):

    try:

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4
            )

        return True

    except Exception as e:

        print(
            "JSON WRITE ERROR:",
            file_path,
            e
        )

        return False


# ============================================================
# RESULT FUNCTIONS
# ============================================================

def get_latest_result():

    result = load_json_file(
        RESULT_FILE,
        DEFAULT_RESULT.copy()
    )

    if not isinstance(
        result,
        dict
    ):

        result = DEFAULT_RESULT.copy()

    # Make sure every field exists
    for key, value in DEFAULT_RESULT.items():

        if key not in result:

            result[key] = value

    return result


def save_latest_result(
    result
):

    return save_json_file(
        RESULT_FILE,
        result
    )


# ============================================================
# HISTORY FUNCTIONS
# ============================================================

def get_history():

    history = load_json_file(
        HISTORY_FILE,
        []
    )

    if not isinstance(
        history,
        list
    ):

        history = []

    return history


def save_history(
    history
):

    return save_json_file(
        HISTORY_FILE,
        history
    )


def add_history_record(
    result
):

    history = get_history()

    history.insert(
        0,
        result
    )

    # Keep latest 100 predictions
    history = history[:100]

    save_history(
        history
    )

    return history


# ============================================================
# ANALYTICS
# ============================================================

def get_analytics():

    history = get_history()

    total = len(
        history
    )

    healthy = 0

    bacterial_spot = 0

    cercospora = 0

    curl_virus = 0

    unclear = 0

    other = 0

    confidence_total = 0.0

    confidence_count = 0


    for record in history:

        disease = str(
            record.get(
                "disease",
                ""
            )
        ).lower()


        # --------------------------------------------
        # DISEASE COUNTS
        # --------------------------------------------

        if "healthy" in disease:

            healthy += 1

        elif "bacterial" in disease:

            bacterial_spot += 1

        elif "cercospora" in disease:

            cercospora += 1

        elif "curl" in disease:

            curl_virus += 1

        elif "unclear" in disease:

            unclear += 1

        else:

            other += 1


        # --------------------------------------------
        # CONFIDENCE
        # --------------------------------------------

        try:

            confidence = float(
                record.get(
                    "confidence",
                    0
                )
            )

            confidence_total += confidence

            confidence_count += 1

        except Exception:

            pass


    if confidence_count > 0:

        average_confidence = (
            confidence_total /
            confidence_count
        )

    else:

        average_confidence = 0


    diseased = (
        bacterial_spot
        +
        cercospora
        +
        curl_virus
    )


    if total > 0:

        disease_percentage = (
            diseased /
            total
        ) * 100

        healthy_percentage = (
            healthy /
            total
        ) * 100

    else:

        disease_percentage = 0

        healthy_percentage = 0


    latest = get_latest_result()


    statistics = {

        # Main values
        "total": total,

        "healthy": healthy,

        "diseased": diseased,

        "unclear": unclear,

        "other": other,

        # Individual diseases
        "bacterial_spot":
            bacterial_spot,

        "cercospora":
            cercospora,

        "cercospora_leaf_spot":
            cercospora,

        "curl_virus":
            curl_virus,

        # Percentages
        "healthy_percentage":
            round(
                healthy_percentage,
                2
            ),

        "disease_percentage":
            round(
                disease_percentage,
                2
            ),

        # Confidence
        "average_confidence":
            round(
                average_confidence,
                2
            ),

        # Latest prediction
        "last_prediction":
            latest.get(
                "disease",
                "No prediction yet"
            ),

        "last_confidence":
            latest.get(
                "confidence",
                0
            ),

        "last_image":
            latest.get(
                "image",
                ""
            ),

        "last_timestamp":
            latest.get(
                "timestamp",
                ""
            )
    }

    return statistics


# ============================================================
# CONTEXT DATA
# ============================================================

def get_dashboard_data():

    result = get_latest_result()

    statistics = get_analytics()

    history = get_history()

    return {

        "result":
            result,

        "detection":
            result,

        "latest_result":
            result,

        "statistics":
            statistics,

        "history":
            history,

        "rover":
            rover_state,

        "settings":
            rover_settings
    }


# ============================================================
# HOME / DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    data = get_dashboard_data()

    return render_template(
        "dashboard.html",
        **data
    )


# ============================================================
# ANALYTICS PAGE
# ============================================================

@app.route("/analytics")
def analytics():

    result = get_latest_result()

    statistics = get_analytics()

    history = get_history()

    return render_template(

        "analytics.html",

        result=result,

        detection=result,

        latest_result=result,

        statistics=statistics,

        history=history,

        rover=rover_state,

        settings=rover_settings
    )


# ============================================================
# DATASET PAGE
# ============================================================

@app.route("/dataset")
def dataset():

    result = get_latest_result()

    statistics = get_analytics()

    history = get_history()

    return render_template(

        "dataset.html",

        result=result,

        detection=result,

        latest_result=result,

        statistics=statistics,

        history=history,

        rover=rover_state,

        settings=rover_settings
    )


# ============================================================
# ROVER PAGE
# ============================================================

@app.route("/rover")
def rover():

    result = get_latest_result()

    statistics = get_analytics()

    return render_template(

        "rover.html",

        result=result,

        detection=result,

        latest_result=result,

        statistics=statistics,

        rover=rover_state,

        rover_state=rover_state,

        settings=rover_settings
    )


# ============================================================
# SETTINGS PAGE
# ============================================================

@app.route("/settings")
def settings():

    result = get_latest_result()

    statistics = get_analytics()

    return render_template(

        "settings.html",

        result=result,

        detection=result,

        latest_result=result,

        statistics=statistics,

        rover=rover_state,

        settings=rover_settings
    )


# ============================================================
# API HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "api":
            "working",

        "service":
            "Precision Agriculture Smart Rover",

        "status":
            "online"
    })


# ============================================================
# API - GET LATEST RESULT
# ============================================================

@app.route(
    "/api/result",
    methods=["GET"]
)
def api_result():

    result = get_latest_result()

    return jsonify({

        "success":
            True,

        "result":
            result
    })


# ============================================================
# API - GET HISTORY
# ============================================================

@app.route(
    "/api/history",
    methods=["GET"]
)
def api_history():

    history = get_history()

    return jsonify({

        "success":
            True,

        "count":
            len(history),

        "history":
            history
    })


# ============================================================
# API - GET ANALYTICS
# ============================================================

@app.route(
    "/api/analytics",
    methods=["GET"]
)
def api_analytics():

    statistics = get_analytics()

    return jsonify({

        "success":
            True,

        "statistics":
            statistics
    })


# ============================================================
# MAIN AI RESULT API
# ============================================================

@app.route(
    "/api/update-result",
    methods=["POST"]
)
def update_result():

    try:

        print("\n")
        print(
            "=============================================="
        )

        print(
            "       NEW AI RESULT RECEIVED"
        )

        print(
            "=============================================="
        )


        # ====================================================
        # GET DATA FROM PREDICT.PY
        # ====================================================

        disease = request.form.get(
            "disease",
            "Unknown"
        )

        chemical = request.form.get(
            "chemical",
            "Not available"
        )

        fertilizer = request.form.get(
            "fertilizer",
            "Not available"
        )

        treatment = request.form.get(
            "treatment",
            "Not available"
        )

        confidence = request.form.get(
            "confidence",
            "0"
        )


        # ====================================================
        # CONVERT CONFIDENCE
        # ====================================================

        try:

            confidence_value = float(
                confidence
            )

        except Exception:

            confidence_value = 0.0


        # Keep confidence between 0 and 100

        confidence_value = max(
            0.0,
            min(
                100.0,
                confidence_value
            )
        )


        # ====================================================
        # GET IMAGE
        # ====================================================

        image = request.files.get(
            "image"
        )


        if image is None:

            print(
                "ERROR: No image received."
            )

            return jsonify({

                "success":
                    False,

                "message":
                    "No image received"
            }), 400


        # ====================================================
        # IMAGE FILENAME
        # ====================================================

        original_filename = (
            image.filename
            or
            "leaf.jpg"
        )


        # Remove unsafe path components

        original_filename = os.path.basename(
            original_filename
        )


        # Generate unique filename

        file_extension = os.path.splitext(
            original_filename
        )[1]


        if not file_extension:

            file_extension = ".jpg"


        filename = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            +
            "_"
            +
            uuid.uuid4().hex[:6]
            +
            file_extension
        )


        # ====================================================
        # SAVE IMAGE
        # ====================================================

        image_path = os.path.join(
            IMAGE_FOLDER,
            filename
        )


        image.save(
            image_path
        )


        print(
            "Image saved:",
            filename
        )


        # ====================================================
        # CREATE RESULT
        # ====================================================

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        result = {

            "image":
                filename,

            "disease":
                disease,

            "chemical":
                chemical,

            "fertilizer":
                fertilizer,

            "treatment":
                treatment,

            "confidence":
                round(
                    confidence_value,
                    2
                ),

            "timestamp":
                timestamp
        }


        # ====================================================
        # SAVE LATEST RESULT
        # ====================================================

        save_latest_result(
            result
        )


        # ====================================================
        # ADD TO HISTORY
        # ====================================================

        add_history_record(
            result
        )


        # ====================================================
        # PRINT RESULT
        # ====================================================

        print("\nAI RESULT:")

        print(
            "Disease:",
            disease
        )

        print(
            "Confidence:",
            f"{confidence_value:.2f}%"
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
            "\nResult saved successfully."
        )


        # ====================================================
        # RETURN JSON
        # ====================================================

        return jsonify({

            "success":
                True,

            "message":
                "AI result and image updated successfully",

            "result":
                result
        })


    except Exception as e:

        print("\n")
        print(
            "=============================================="
        )

        print(
            "              API ERROR"
        )

        print(
            "=============================================="
        )

        print(
            str(e)
        )


        return jsonify({

            "success":
                False,

            "message":
                str(e)
        }), 500


# ============================================================
# ROVER CONTROL API
# ============================================================

@app.route(
    "/api/rover/control",
    methods=["POST"]
)
def rover_control():

    try:

        data = request.get_json(
            silent=True
        )

        if data is None:

            data = request.form


        command = str(
            data.get(
                "command",
                "stop"
            )
        ).lower()


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        if command == "forward":

            rover_state[
                "direction"
            ] = "Forward"


        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        elif command == "backward":

            rover_state[
                "direction"
            ] = "Backward"


        # ----------------------------------------------------
        # LEFT
        # ----------------------------------------------------

        elif command == "left":

            rover_state[
                "direction"
            ] = "Left"


        # ----------------------------------------------------
        # RIGHT
        # ----------------------------------------------------

        elif command == "right":

            rover_state[
                "direction"
            ] = "Right"


        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        elif command == "stop":

            rover_state[
                "direction"
            ] = "Stopped"


        # ----------------------------------------------------
        # PUMP ON
        # ----------------------------------------------------

        elif command == "pump_on":

            rover_state[
                "pump"
            ] = True


        # ----------------------------------------------------
        # PUMP OFF
        # ----------------------------------------------------

        elif command == "pump_off":

            rover_state[
                "pump"
            ] = False


        # ----------------------------------------------------
        # CONNECT
        # ----------------------------------------------------

        elif command == "connect":

            rover_state[
                "connected"
            ] = True


        # ----------------------------------------------------
        # DISCONNECT
        # ----------------------------------------------------

        elif command == "disconnect":

            rover_state[
                "connected"
            ] = False

            rover_state[
                "direction"
            ] = "Stopped"


        # ----------------------------------------------------
        # UNKNOWN COMMAND
        # ----------------------------------------------------

        else:

            return jsonify({

                "success":
                    False,

                "message":
                    "Unknown rover command",

                "command":
                    command
            }), 400


        print(
            "Rover command:",
            command
        )


        return jsonify({

            "success":
                True,

            "message":
                "Rover command received",

            "command":
                command,

            "rover":
                rover_state
        })


    except Exception as e:

        return jsonify({

            "success":
                False,

            "message":
                str(e)
        }), 500


# ============================================================
# ROVER STATUS API
# ============================================================

@app.route(
    "/api/rover/status",
    methods=["GET"]
)
def rover_status():

    return jsonify({

        "success":
            True,

        "rover":
            rover_state
    })


# ============================================================
# ROVER SPEED API
# ============================================================

@app.route(
    "/api/rover/speed",
    methods=["POST"]
)
def rover_speed():

    try:

        data = request.get_json(
            silent=True
        )

        if data is None:

            data = request.form


        speed = int(
            data.get(
                "speed",
                50
            )
        )


        speed = max(
            0,
            min(
                100,
                speed
            )
        )


        rover_state[
            "speed"
        ] = speed


        return jsonify({

            "success":
                True,

            "speed":
                speed,

            "rover":
                rover_state
        })


    except Exception as e:

        return jsonify({

            "success":
                False,

            "message":
                str(e)
        }), 500


# ============================================================
# PUMP API
# ============================================================

@app.route(
    "/api/rover/pump",
    methods=["POST"]
)
def rover_pump():

    try:

        data = request.get_json(
            silent=True
        )

        if data is None:

            data = request.form


        state = data.get(
            "state",
            False
        )


        if isinstance(
            state,
            str
        ):

            state = state.lower() in [
                "true",
                "1",
                "on",
                "yes"
            ]


        rover_state[
            "pump"
        ] = bool(
            state
        )


        return jsonify({

            "success":
                True,

            "pump":
                rover_state["pump"]
        })


    except Exception as e:

        return jsonify({

            "success":
                False,

            "message":
                str(e)
        }), 500


# ============================================================
# SETTINGS API
# ============================================================

@app.route(
    "/api/settings",
    methods=["GET"]
)
def get_settings():

    return jsonify({

        "success":
            True,

        "settings":
            rover_settings
    })


# ============================================================
# UPDATE SETTINGS
# ============================================================

@app.route(
    "/api/settings",
    methods=["POST"]
)
def update_settings():

    try:

        data = request.get_json(
            silent=True
        )

        if data is None:

            data = request.form


        # --------------------------------------------
        # AUTO MODE
        # --------------------------------------------

        if "auto_mode" in data:

            value = data.get(
                "auto_mode"
            )

            if isinstance(
                value,
                str
            ):

                value = value.lower() in [
                    "true",
                    "1",
                    "yes",
                    "on"
                ]

            rover_settings[
                "auto_mode"
            ] = bool(
                value
            )


        # --------------------------------------------
        # SPRAY
        # --------------------------------------------

        if "spray_enabled" in data:

            value = data.get(
                "spray_enabled"
            )

            if isinstance(
                value,
                str
            ):

                value = value.lower() in [
                    "true",
                    "1",
                    "yes",
                    "on"
                ]

            rover_settings[
                "spray_enabled"
            ] = bool(
                value
            )


        # --------------------------------------------
        # DETECTION
        # --------------------------------------------

        if "detection_enabled" in data:

            value = data.get(
                "detection_enabled"
            )

            if isinstance(
                value,
                str
            ):

                value = value.lower() in [
                    "true",
                    "1",
                    "yes",
                    "on"
                ]

            rover_settings[
                "detection_enabled"
            ] = bool(
                value
            )


        # --------------------------------------------
        # CONFIDENCE THRESHOLD
        # --------------------------------------------

        if "confidence_threshold" in data:

            try:

                threshold = float(
                    data.get(
                        "confidence_threshold"
                    )
                )

                threshold = max(
                    0,
                    min(
                        100,
                        threshold
                    )
                )

                rover_settings[
                    "confidence_threshold"
                ] = threshold

            except Exception:

                pass


        # --------------------------------------------
        # SPEED
        # --------------------------------------------

        if "speed" in data:

            try:

                speed = int(
                    data.get(
                        "speed"
                    )
                )

                speed = max(
                    0,
                    min(
                        100,
                        speed
                    )
                )

                rover_settings[
                    "speed"
                ] = speed

            except Exception:

                pass


        return jsonify({

            "success":
                True,

            "message":
                "Settings updated",

            "settings":
                rover_settings
        })


    except Exception as e:

        return jsonify({

            "success":
                False,

            "message":
                str(e)
        }), 500


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route(
    "/api/history/clear",
    methods=["POST"]
)
def clear_history():

    try:

        save_history(
            []
        )

        return jsonify({

            "success":
                True,

            "message":
                "Prediction history cleared"
        })

    except Exception as e:

        return jsonify({

            "success":
                False,

            "message":
                str(e)
        }), 500


# ============================================================
# RESET LATEST RESULT
# ============================================================

@app.route(
    "/api/result/clear",
    methods=["POST"]
)
def clear_result():

    try:

        save_latest_result(
            DEFAULT_RESULT.copy()
        )

        return jsonify({

            "success":
                True,

            "message":
                "Latest result cleared"
        })

    except Exception as e:

        return jsonify({

            "success":
                False,

            "message":
                str(e)
        }), 500


# ============================================================
# FAVICON
# ============================================================

@app.route(
    "/favicon.ico"
)
def favicon():

    return "", 204


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({

        "success":
            False,

        "error":
            "Page not found"
    }), 404


@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({

        "success":
            False,

        "error":
            "Internal server error"
    }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        "=============================================="
    )

    print(
        "PRECISION AGRICULTURE SMART ROVER"
    )

    print(
        "Flask server starting..."
    )

    print(
        "Port:",
        port
    )

    print(
        "=============================================="
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False
    )