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
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static",
    template_folder="templates"
)

app.config["JSON_SORT_KEYS"] = False


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
IMAGE_FOLDER = os.path.join(STATIC_FOLDER, "images")

RESULT_FILE = os.path.join(BASE_DIR, "result.json")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

app.template_folder = TEMPLATE_FOLDER
app.static_folder = STATIC_FOLDER

os.makedirs(IMAGE_FOLDER, exist_ok=True)


# ============================================================
# NO CACHE
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
# DEFAULT AI RESULT
# ============================================================

DEFAULT_RESULT = {
    "result_id": "",
    "image": "",
    "disease": "No prediction yet",
    "chemical": "Not available",
    "fertilizer": "Not available",
    "treatment": "No prediction has been processed yet.",
    "confidence": 0.0,
    "timestamp": ""
}


# ============================================================
# ROVER STATE
# ============================================================

rover_state = {

    "connected": False,

    "direction": "Stopped",

    "speed": 50,

    "pump": False,

    "mode": "Manual",

    "last_command": "stop",

    "command_id": 0,

    "last_seen": None
}


# ============================================================
# ROVER SETTINGS
# ============================================================

rover_settings = {

    "auto_mode": False,

    "spray_enabled": True,

    "detection_enabled": True,

    "confidence_threshold": 80,

    "speed": 50
}


# ============================================================
# GPIO PIN MAPPINGS (Raspberry Pi 4 B - Smart Rover)
# ============================================================

GPIO_MAPPING = {
    "left_motor": {
        "title": "Left Motor Driver (L298N Channel A)",
        "in1": {"bcm": 17, "pin": 11, "name": "LEFT_IN1", "desc": "Forward Direction"},
        "in2": {"bcm": 27, "pin": 13, "name": "LEFT_IN2", "desc": "Reverse Direction"},
        "ena": {"bcm": 12, "pin": 32, "name": "LEFT_EN", "desc": "PWM Speed (PWM0)"}
    },
    "right_motor": {
        "title": "Right Motor Driver (L298N Channel B)",
        "in3": {"bcm": 22, "pin": 15, "name": "RIGHT_IN1", "desc": "Forward Direction"},
        "in4": {"bcm": 23, "pin": 16, "name": "RIGHT_IN2", "desc": "Reverse Direction"},
        "enb": {"bcm": 13, "pin": 33, "name": "RIGHT_EN", "desc": "PWM Speed (PWM1)"}
    },
    "spray_pump": {
        "title": "Pesticide Spray Pump (5V Relay)",
        "pump_pin": {"bcm": 26, "pin": 37, "name": "PUMP_PIN", "desc": "Relay Trigger (IN)"}
    },
    "ultrasonic": {
        "title": "Obstacle Sensor (HC-SR04 Ultrasonic)",
        "trig": {"bcm": 24, "pin": 18, "name": "TRIG_PIN", "desc": "Trigger 10us Pulse"},
        "echo": {"bcm": 25, "pin": 22, "name": "ECHO_PIN", "desc": "Echo Input (via 1k/2k Divider)"}
    },
    "camera": {
        "title": "CSI Camera Sensor",
        "interface": {"bcm": "CSI-2", "pin": "15-Pin Ribbon", "name": "CAM_PORT", "desc": "OmniVision OV5647 / Sony IMX708"}
    }
}


# ============================================================
# JSON HELPERS
# ============================================================

def load_json_file(file_path, default_value):

    try:

        if not os.path.exists(file_path):
            return default_value

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print("JSON READ ERROR:", e)

        return default_value


def save_json_file(file_path, data):

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

        print("JSON WRITE ERROR:", e)

        return False


# ============================================================
# RESULT
# ============================================================

def get_latest_result():

    result = load_json_file(
        RESULT_FILE,
        DEFAULT_RESULT.copy()
    )

    if not isinstance(result, dict):
        result = DEFAULT_RESULT.copy()

    for key, value in DEFAULT_RESULT.items():

        if key not in result:
            result[key] = value

    return result


def save_latest_result(result):

    return save_json_file(
        RESULT_FILE,
        result
    )


# ============================================================
# HISTORY
# ============================================================

def get_history():

    history = load_json_file(
        HISTORY_FILE,
        []
    )

    if not isinstance(history, list):
        history = []

    return history


def save_history(history):

    return save_json_file(
        HISTORY_FILE,
        history
    )


def add_history_record(result):

    history = get_history()

    history.insert(
        0,
        result
    )

    history = history[:100]

    save_history(history)

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


    if confidence_count:

        average_confidence = (
            confidence_total /
            confidence_count
        )

    else:

        average_confidence = 0


    diseased = (
        bacterial_spot
        + cercospora
        + curl_virus
    )


    if total:

        disease_percentage = (
            diseased / total
        ) * 100

        healthy_percentage = (
            healthy / total
        ) * 100

    else:

        disease_percentage = 0
        healthy_percentage = 0


    latest = get_latest_result()


    return {

        "total": total,

        "healthy": healthy,

        "diseased": diseased,

        "unclear": unclear,

        "other": other,

        "bacterial_spot": bacterial_spot,

        "cercospora": cercospora,

        "cercospora_leaf_spot": cercospora,

        "curl_virus": curl_virus,

        "healthy_percentage":
            round(healthy_percentage, 2),

        "disease_percentage":
            round(disease_percentage, 2),

        "average_confidence":
            round(average_confidence, 2),

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

    return {

        "result": result,

        "detection": result,

        "latest_result": result,

        "statistics": get_analytics(),

        "history": get_history(),

        "rover": rover_state,

        "settings": rover_settings

    }


# ============================================================
# WEB PAGES
# ============================================================

@app.route("/")
def dashboard():

    return render_template(
        "dashboard.html",
        **get_dashboard_data()
    )


@app.route("/analytics")
def analytics():

    result = get_latest_result()

    return render_template(
        "analytics.html",
        result=result,
        detection=result,
        latest_result=result,
        statistics=get_analytics(),
        history=get_history(),
        rover=rover_state,
        settings=rover_settings
    )


@app.route("/dataset")
def dataset():

    result = get_latest_result()

    return render_template(
        "dataset.html",
        result=result,
        detection=result,
        latest_result=result,
        statistics=get_analytics(),
        history=get_history(),
        rover=rover_state,
        settings=rover_settings
    )


@app.route("/rover")
def rover():

    result = get_latest_result()

    return render_template(
        "rover.html",
        result=result,
        detection=result,
        latest_result=result,
        statistics=get_analytics(),
        rover=rover_state,
        rover_state=rover_state,
        settings=rover_settings,
        gpio_mapping=GPIO_MAPPING
    )


@app.route("/settings")
def settings():

    result = get_latest_result()

    return render_template(
        "settings.html",
        result=result,
        detection=result,
        latest_result=result,
        statistics=get_analytics(),
        rover=rover_state,
        settings=rover_settings,
        gpio_mapping=GPIO_MAPPING
    )


# ============================================================
# PWA ICON
# ============================================================

@app.route("/static/icons/<path:filename>")
def pwa_icon(filename):

    return send_from_directory(
        os.path.join(
            STATIC_FOLDER,
            "icons"
        ),
        filename
    )


# ============================================================
# MANIFEST
# ============================================================

@app.route("/manifest.json")
def manifest():

    manifest_path = os.path.join(
        STATIC_FOLDER,
        "manifest.json"
    )

    if not os.path.exists(manifest_path):

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


    response = make_response(data)

    response.headers[
        "Content-Type"
    ] = "application/manifest+json"

    return response


# ============================================================
# SERVICE WORKER
# ============================================================

@app.route("/service-worker.js")
def service_worker():

    path = os.path.join(
        STATIC_FOLDER,
        "service-worker.js"
    )

    if not os.path.exists(path):

        return (
            "Service worker not found.",
            404
        )


    with open(
        path,
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
        "Service-Worker-Allowed"
    ] = "/"

    return response


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "api": "working",

        "service":
            "Precision Agriculture Smart Rover",

        "status": "online"

    })


# ============================================================
# RESULT API
# ============================================================

@app.route("/api/result", methods=["GET"])
def api_result():

    return jsonify({

        "success": True,

        "result":
            get_latest_result()

    })


# ============================================================
# DASHBOARD STATE API
# ============================================================

@app.route(
    "/api/dashboard-state",
    methods=["GET"]
)
def api_dashboard_state():

    response = jsonify({

        "success": True,

        "result":
            get_latest_result(),

        "statistics":
            get_analytics(),

        "rover":
            rover_state,

        "settings":
            rover_settings,

        "server_time":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    })

    return response


# ============================================================
# HISTORY API
# ============================================================

@app.route("/api/history", methods=["GET"])
def api_history():

    history = get_history()

    return jsonify({

        "success": True,

        "count":
            len(history),

        "history":
            history

    })


# ============================================================
# ANALYTICS API
# ============================================================

@app.route("/api/analytics", methods=["GET"])
def api_analytics():

    return jsonify({

        "success": True,

        "statistics":
            get_analytics()

    })


# ============================================================
# CAMERA AI RESULT
# DO NOT CHANGE THIS URL
# ============================================================

@app.route(
    "/api/update-result",
    methods=["POST"]
)
def update_result():

    try:

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


        image = request.files.get(
            "image"
        )


        if image is None:

            return jsonify({

                "success": False,

                "message":
                    "No image received"

            }), 400


        original_filename = os.path.basename(
            image.filename or "leaf.jpg"
        )


        extension = os.path.splitext(
            original_filename
        )[1]


        if not extension:
            extension = ".jpg"


        filename = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + "_"
            + uuid.uuid4().hex[:8]
            + extension
        )


        image.save(
            os.path.join(
                IMAGE_FOLDER,
                filename
            )
        )


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
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

        }


        save_latest_result(result)

        add_history_record(result)


        print(
            "AI RESULT:",
            disease,
            confidence_value
        )


        return jsonify({

            "success": True,

            "message":
                "AI result and image updated successfully",

            "result":
                result

        })


    except Exception as e:

        print(
            "AI API ERROR:",
            e
        )

        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 500


# ============================================================
# ROVER CONTROL
#
# forward
# backward
# left
# right
# stop
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
        ).lower().strip()


        valid_commands = [

            "forward",
            "backward",
            "left",
            "right",
            "stop"

        ]


        if command not in valid_commands:

            return jsonify({

                "success": False,

                "message":
                    "Invalid rover command",

                "command":
                    command

            }), 400


        direction_names = {

            "forward": "Forward",

            "backward": "Backward",

            "left": "Left",

            "right": "Right",

            "stop": "Stopped"

        }


        rover_state[
            "direction"
        ] = direction_names[command]


        rover_state[
            "last_command"
        ] = command


        rover_state[
            "command_id"
        ] += 1


        if command == "stop":

            rover_state["speed"] = 0

        elif rover_state["speed"] == 0:

            rover_state["speed"] = \
                rover_settings["speed"]


        print(
            "ROVER COMMAND:",
            command
        )


        return jsonify({

            "success": True,

            "message":
                "Rover command received",

            "command":
                command,

            "command_id":
                rover_state["command_id"],

            "rover":
                rover_state

        })


    except Exception as e:

        print(
            "ROVER CONTROL ERROR:",
            e
        )

        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 500


# ============================================================
# RASPBERRY PI COMMAND API
#
# Raspberry Pi will call this repeatedly.
# ============================================================

@app.route(
    "/api/rover/command",
    methods=["GET"]
)
def rover_command():

    return jsonify({

        "success": True,

        "command":
            rover_state["last_command"],

        "command_id":
            rover_state["command_id"],

        "direction":
            rover_state["direction"],

        "speed":
            rover_state["speed"],

        "pump":
            rover_state["pump"],

        "mode":
            rover_state["mode"]

    })


# ============================================================
# RASPBERRY PI HEARTBEAT
# ============================================================

@app.route(
    "/api/rover/heartbeat",
    methods=["POST"]
)
def rover_heartbeat():

    try:

        rover_state["connected"] = True

        rover_state["last_seen"] = \
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )


        return jsonify({

            "success": True,

            "message":
                "Raspberry Pi connected",

            "rover":
                rover_state

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 500


# ============================================================
# GPIO PIN CONFIGURATION API
# ============================================================

@app.route(
    "/api/rover/gpio-config",
    methods=["GET"]
)
def rover_gpio_config():

    return jsonify({
        "success": True,
        "board": "Raspberry Pi 4 Model B",
        "gpio_mapping": GPIO_MAPPING
    })


# ============================================================
# ROVER STATUS
# ============================================================

@app.route(
    "/api/rover/status",
    methods=["GET"]
)
def rover_status():

    return jsonify({

        "success": True,

        "rover":
            rover_state,

        "server_time":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

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


        rover_state["speed"] = speed

        rover_settings["speed"] = speed


        return jsonify({

            "success": True,

            "speed":
                speed,

            "rover":
                rover_state

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 500


# ============================================================
# PUMP CONTROL
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


        if isinstance(state, str):

            state = state.lower() in [

                "true",
                "1",
                "on",
                "yes"

            ]


        state = bool(state)


        rover_state["pump"] = state


        rover_state["last_command"] = (

            "pump_on"
            if state
            else
            "pump_off"

        )


        rover_state["command_id"] += 1


        return jsonify({

            "success": True,

            "message":
                "Pump command received",

            "pump":
                state,

            "command":
                rover_state["last_command"],

            "command_id":
                rover_state["command_id"]

        })


    except Exception as e:

        return jsonify({

            "success": False,

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

        "success": True,

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


        boolean_fields = [

            "auto_mode",
            "spray_enabled",
            "detection_enabled"

        ]


        for field in boolean_fields:

            if field in data:

                value = data.get(field)


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


                rover_settings[field] = bool(
                    value
                )


        if "confidence_threshold" in data:

            try:

                threshold = float(
                    data.get(
                        "confidence_threshold"
                    )
                )

                rover_settings[
                    "confidence_threshold"
                ] = max(
                    0,
                    min(
                        100,
                        threshold
                    )
                )

            except Exception:

                pass


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

                rover_settings["speed"] = speed
                rover_state["speed"] = speed

            except Exception:

                pass


        return jsonify({

            "success": True,

            "message":
                "Settings updated",

            "settings":
                rover_settings

        })


    except Exception as e:

        return jsonify({

            "success": False,

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

        save_history([])

        return jsonify({

            "success": True,

            "message":
                "Prediction history cleared"

        })

    except Exception as e:

        return jsonify({

            "success": False,

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

            "success": True,

            "message":
                "Latest result cleared"

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 500


# ============================================================
# FAVICON
# ============================================================

@app.route("/favicon.ico")
def favicon():

    return "", 204


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({

        "success": False,

        "error":
            "Page not found"

    }), 404


@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({

        "success": False,

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
        "CAMERA AI:"
    )

    print(
        "/api/update-result"
    )

    print(
        "ROVER CONTROL:"
    )

    print(
        "/api/rover/control"
    )

    print(
        "RASPBERRY PI COMMAND:"
    )

    print(
        "/api/rover/command"
    )

    print(
        "RASPBERRY PI HEARTBEAT:"
    )

    print(
        "/api/rover/heartbeat"
    )

    print(
        "PWA:"
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