# ============================================================
# PRECISION AGRICULTURE SMART ROVER
# COMPLETE FLASK WEBSITE + PWA BACKEND
#
# CAMERA AI
#     ↓
# /api/update-result
#     ↓
# Flask
#     ↓
# result.json + history.json + image
#     ↓
# WEBSITE + PWA APP
#
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
    make_response,
    send_from_directory
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static",
    template_folder="templates"
)

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
# FLASK CONFIGURATION
# ============================================================

app.template_folder = TEMPLATE_FOLDER
app.static_folder = STATIC_FOLDER

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)


# ============================================================
# GLOBAL NO-CACHE HEADERS
#
# Important:
# API and live dashboard data must never be cached.
# ============================================================

@app.after_request
def add_no_cache_headers(response):

    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, "
        "max-age=0, s-maxage=0"
    )

    response.headers["Pragma"] = "no-cache"

    response.headers["Expires"] = "0"

    return response


# ============================================================
# DEFAULT RESULT
# ============================================================

DEFAULT_RESULT = {

    "result_id": "",

    "image": "",

    "disease": "No prediction yet",

    "chemical": "Not available",

    "fertilizer": "Not available",

    "treatment":
        "No prediction has been processed yet.",

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

            data = json.load(f)

        return data

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

    total = len(history)

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


        # ----------------------------------------------------
        # DISEASE COUNTS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

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


    return {

        "total":
            total,

        "healthy":
            healthy,

        "diseased":
            diseased,

        "unclear":
            unclear,

        "other":
            other,

        "bacterial_spot":
            bacterial_spot,

        "cercospora":
            cercospora,

        "cercospora_leaf_spot":
            cercospora,

        "curl_virus":
            curl_virus,

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

        "average_confidence":
            round(
                average_confidence,
                2
            ),

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


# ============================================================
# DASHBOARD DATA
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
# PWA STATIC ICONS
# ============================================================

@app.route("/static/icons/<path:filename>")
def pwa_icon(filename):
    return send_from_directory(
        os.path.join(STATIC_FOLDER, "icons"),
        filename
    )


# ============================================================
# PWA MANIFEST
#
# If manifest.json is inside static/,
# this route makes sure it is served correctly.
# ============================================================

@app.route(
    "/manifest.json"
)
def manifest():

    manifest_path = os.path.join(
        STATIC_FOLDER,
        "manifest.json"
    )

    if not os.path.exists(
        manifest_path
    ):

        return jsonify({

            "name":
                "Precision Agriculture Smart Rover",

            "short_name":
                "Smart Rover",

            "start_url":
                "/",

            "display":
                "standalone",

            "background_color":
                "#f4f7f5",

            "theme_color":
                "#12372a"

        })


    with open(
        manifest_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = f.read()


    response = make_response(
        data
    )

    response.headers[
        "Content-Type"
    ] = "application/manifest+json"

    response.headers[
        "Cache-Control"
    ] = "no-cache, no-store, must-revalidate"

    return response


# ============================================================
# SERVICE WORKER
#
# IMPORTANT:
# Service worker must be served from the ROOT.
#
# /service-worker.js
#
# This gives it control over the complete website/app.
# ============================================================

@app.route(
    "/service-worker.js"
)
def service_worker():

    service_worker_path = os.path.join(
        STATIC_FOLDER,
        "service-worker.js"
    )


    if not os.path.exists(
        service_worker_path
    ):

        return (
            "Service worker not found.",
            404
        )


    with open(
        service_worker_path,
        "r",
        encoding="utf-8"
    ) as f:

        javascript = f.read()


    response = make_response(
        javascript
    )


    response.headers[
        "Content-Type"
    ] = "application/javascript"


    response.headers[
        "Cache-Control"
    ] = (
        "no-cache, no-store, "
        "must-revalidate"
    )


    response.headers[
        "Service-Worker-Allowed"
    ] = "/"


    return response


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
# API HEALTH
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
# API RESULT
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
# LIVE DASHBOARD API
#
# Website and app both use this.
#
# CAMERA AI → /api/update-result
# APP       → /api/dashboard-state
# ============================================================

@app.route(
    "/api/dashboard-state",
    methods=["GET"]
)
def api_dashboard_state():

    result = get_latest_result()

    statistics = get_analytics()


    response = jsonify({

        "success":
            True,

        "result":
            result,

        "statistics":
            statistics,

        "rover":
            rover_state,

        "settings":
            rover_settings,

        "server_time":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    })


    response.headers[
        "Cache-Control"
    ] = (
        "no-store, no-cache, "
        "must-revalidate, "
        "max-age=0"
    )


    response.headers[
        "Pragma"
    ] = "no-cache"


    response.headers[
        "Expires"
    ] = "0"


    return response


# ============================================================
# HISTORY API
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
# ANALYTICS API
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
# MAIN CAMERA AI RESULT API
#
# YOUR CAMERA PYTHON CODE USES THIS.
#
# DO NOT CHANGE THE URL.
# ============================================================

@app.route(
    "/api/update-result",
    methods=["POST"]
)
def update_result():

    try:

        print()
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
        # RECEIVE AI DATA
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
        # CONFIDENCE
        # ====================================================

        try:

            confidence_value = float(
                confidence
            )

        except Exception:

            confidence_value = 0.0


        confidence_value = max(
            0.0,
            min(
                100.0,
                confidence_value
            )
        )


        # ====================================================
        # RECEIVE CAMERA IMAGE
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
        # SAFE ORIGINAL FILENAME
        # ====================================================

        original_filename = (
            image.filename
            or
            "leaf.jpg"
        )


        original_filename = os.path.basename(
            original_filename
        )


        # ====================================================
        # FILE EXTENSION
        # ====================================================

        file_extension = os.path.splitext(
            original_filename
        )[1]


        if not file_extension:

            file_extension = ".jpg"


        # ====================================================
        # UNIQUE IMAGE NAME
        # ====================================================

        filename = (

            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            + "_"

            + uuid.uuid4().hex[:8]

            + file_extension

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
        # TIMESTAMP
        # ====================================================

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        # ====================================================
        # CREATE RESULT
        # ====================================================

        result = {

            "result_id":
                uuid.uuid4().hex,

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
        # SAVE HISTORY
        # ====================================================

        add_history_record(
            result
        )


        # ====================================================
        # SERVER LOG
        # ====================================================

        print()

        print(
            "AI RESULT:"
        )

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

        print()

        print(
            "Result saved successfully."
        )


        # ====================================================
        # RETURN
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

        print()
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
# ROVER CONTROL
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


        if command == "forward":

            rover_state[
                "direction"
            ] = "Forward"


        elif command == "backward":

            rover_state[
                "direction"
            ] = "Backward"


        elif command == "left":

            rover_state[
                "direction"
            ] = "Left"


        elif command == "right":

            rover_state[
                "direction"
            ] = "Right"


        elif command == "stop":

            rover_state[
                "direction"
            ] = "Stopped"


        elif command == "pump_on":

            rover_state[
                "pump"
            ] = True


        elif command == "pump_off":

            rover_state[
                "pump"
            ] = False


        elif command == "connect":

            rover_state[
                "connected"
            ] = True


        elif command == "disconnect":

            rover_state[
                "connected"
            ] = False

            rover_state[
                "direction"
            ] = "Stopped"


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
# ROVER STATUS
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
# ROVER SPEED
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
# PUMP
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
                rover_state[
                    "pump"
                ]

        })


    except Exception as e:

        return jsonify({

            "success":
                False,

            "message":
                str(e)

        }), 500


# ============================================================
# SETTINGS GET
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
# SETTINGS UPDATE
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


        # ----------------------------------------------------
        # AUTO MODE
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # SPRAY
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # DETECTION
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

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
# CLEAR RESULT
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
        "FLASK SERVER STARTING..."
    )

    print(
        "PORT:",
        port
    )

    print(
        "CAMERA AI API:"
    )

    print(
        "/api/update-result"
    )

    print(
        "LIVE DASHBOARD API:"
    )

    print(
        "/api/dashboard-state"
    )

    print(
        "PWA SERVICE WORKER:"
    )

    print(
        "/service-worker.js"
    )

    print(
        "=============================================="
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )