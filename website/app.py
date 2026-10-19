import os
import json
import uuid
import time
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    make_response,
    send_from_directory,
    Response,
    redirect
)
import db


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

PLACEHOLDER_JPEG = (
    b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00'
    b'\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19'
    b'\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9'
    b'=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00'
    b'\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03'
    b'\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9'
)


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
    "pi_online": False,
    "model_online": False,
    "device": "Raspberry Pi 4 Model B",
    "ai_status": "Standby",
    "model_name": "MobileNetV2 (99% Acc) Chili Disease AI",
    "fps": 0.0,
    "last_detection": "NO LEAF DETECTED",
    "last_confidence": 0.0,
    "direction": "Stopped",
    "speed": 50,
    "pump": False,
    "mode": "Manual",
    "last_command": "stop",
    "command_id": 0,
    "pi_ip": "",
    "stream_url": "",
    "disease_name": "NO LEAF DETECTED",
    "disease_chemical": "",
    "disease_fertilizer": "",
    "disease_treatment": "",
    "last_seen": None,
    "last_seen_epoch": 0.0,
    "relay_active_low": True
}


def refresh_rover_connection_state():
    """Dynamically determine if the Raspberry Pi and AI Model are online based on heartbeat timestamp."""
    epoch = rover_state.get("last_seen_epoch", 0.0)
    now = time.time()
    # Heartbeat is sent every 4s by the Pi; if seen within 15 seconds, consider online
    is_alive = bool(epoch and (now - epoch < 15.0))
    rover_state["connected"] = is_alive
    rover_state["pi_online"] = is_alive
    if not is_alive:
        rover_state["model_online"] = False
        rover_state["ai_status"] = "Offline / Standby"
    return is_alive


# ============================================================
# ROVER SETTINGS
# ============================================================

rover_settings = {

    "auto_mode": False,

    "spray_enabled": True,

    "detection_enabled": True,

    "confidence_threshold": 80,

    "speed": 50,

    "invert_right_motor": True,

    "invert_left_motor": False
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
    try:
        latest = db.get_latest_detection()
        if latest and latest.get("disease") and latest.get("disease") != "No prediction yet":
            return latest
    except Exception:
        pass

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
    try:
        db.save_detection(result)
    except Exception as e:
        print("[DB SAVE WARNING]", e)

    return save_json_file(
        RESULT_FILE,
        result
    )


# ============================================================
# HISTORY
# ============================================================

def get_history():
    try:
        hist = db.get_recent_history(limit=100)
        if hist and len(hist) > 0:
            return hist
    except Exception:
        pass

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
    try:
        db.save_detection(result)
    except Exception as e:
        print("[DB ADD HISTORY WARNING]", e)

    history = get_history()
    save_history(history)
    return history


# ============================================================
# ANALYTICS
_cached_analytics = None
_cached_analytics_time = 0.0

def get_analytics(force_refresh=False):
    global _cached_analytics, _cached_analytics_time
    now = time.time()
    if not force_refresh and _cached_analytics is not None and (now - _cached_analytics_time < 2.5):
        return _cached_analytics

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


    analytics_data = {

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

    _cached_analytics = analytics_data
    _cached_analytics_time = now
    return analytics_data


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


@app.route("/daily-data")
@app.route("/daily_data")
def daily_data():

    result = get_latest_result()
    daily_list = db.get_daily_grouped_data()
    stats = db.get_analytics_summary()

    return render_template(
        "daily_data.html",
        daily_data=daily_list,
        statistics=stats,
        result=result,
        detection=result,
        latest_result=result,
        rover=rover_state,
        settings=rover_settings
    )


@app.route("/api/daily-data", methods=["GET"])
def api_daily_data():

    return jsonify({
        "success": True,
        "daily_data": db.get_daily_grouped_data(),
        "statistics": db.get_analytics_summary()
    })


@app.route("/api/daily-data/clear", methods=["POST"])
def api_clear_daily_data():

    success = db.clear_all_detections()

    return jsonify({
        "success": success,
        "message": "Daily detection records cleared successfully." if success else "Error clearing database."
    })


# Global in-memory cloud frame buffer for remote viewing
latest_cloud_frame = None
latest_cloud_frame_time = 0.0


@app.route("/api/camera/stream_frame", methods=["POST"])
def api_camera_stream_frame():
    """Receives live JPEG frame from Raspberry Pi to relay to cloud clients across Gunicorn workers."""
    global latest_cloud_frame, latest_cloud_frame_time
    file = request.files.get("frame")
    data = None
    if file:
        data = file.read()
    elif request.data:
        data = request.data

    if data and len(data) > 100:
        latest_cloud_frame = data
        latest_cloud_frame_time = time.time()

        # Fast write to RAM disk (/dev/shm) if available on Linux container
        shm_path = "/dev/shm/latest_cloud_frame.jpg"
        try:
            with open(shm_path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(shm_path + ".tmp", shm_path)
        except Exception:
            try:
                with open(shm_path, "wb") as f:
                    f.write(data)
            except Exception:
                pass

        # Also write to disk backup for worker synchronization
        try:
            live_path = os.path.join(IMAGE_FOLDER, "latest_live_frame.jpg")
            with open(live_path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(live_path + ".tmp", live_path)
        except Exception:
            pass

        return jsonify({"success": True})
    return jsonify({"success": False, "error": "No frame data"}), 400


@app.route("/video_feed")
def video_feed():
    """Live camera stream reading from shared RAM/disk for high-speed continuous streaming."""
    def generate_stream():
        boundary = b'--FRAME\r\n'
        last_frame = None
        last_sent = None
        last_sent_t = 0.0
        while True:
            frame = None
            # 1. Local RPi RAM disk (Direct on Raspberry Pi)
            if os.path.exists("/dev/shm/rover_stream.jpg"):
                try:
                    with open("/dev/shm/rover_stream.jpg", "rb") as f:
                        frame = f.read()
                except Exception:
                    pass
            # 2. Shared Cloud RAM disk (Fastest across all Gunicorn workers on Render)
            if not frame and os.path.exists("/dev/shm/latest_cloud_frame.jpg"):
                try:
                    with open("/dev/shm/latest_cloud_frame.jpg", "rb") as f:
                        frame = f.read()
                except Exception:
                    pass
            # 3. In-memory cloud frame buffer
            if not frame and latest_cloud_frame is not None and len(latest_cloud_frame) > 100:
                frame = latest_cloud_frame
            # 4. Disk backup file
            if not frame:
                p = os.path.join(IMAGE_FOLDER, "latest_live_frame.jpg")
                if os.path.exists(p):
                    try:
                        with open(p, "rb") as f:
                            frame = f.read()
                    except Exception:
                        pass

            to_send = frame if (frame and len(frame) > 100) else (last_frame if last_frame else PLACEHOLDER_JPEG)
            last_frame = to_send

            now_t = time.time()
            if to_send != last_sent or (now_t - last_sent_t >= 0.35):
                last_sent = to_send
                last_sent_t = now_t
                yield (boundary +
                       b'Content-Type: image/jpeg\r\n' +
                       b'Content-Length: ' + str(len(to_send)).encode() + b'\r\n\r\n' +
                       to_send + b'\r\n')
            time.sleep(0.012)

    resp = Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=FRAME')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route("/api/camera/live_frame")
def camera_live_frame():
    """Instantaneous single live frame for high-speed dynamic refresh across all Gunicorn workers."""
    # 1. Local RPi RAM disk
    if os.path.exists("/dev/shm/rover_stream.jpg"):
        try:
            with open("/dev/shm/rover_stream.jpg", "rb") as f:
                data = f.read()
                if len(data) > 100:
                    resp = Response(data, mimetype='image/jpeg')
                    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
                    resp.headers['Pragma'] = 'no-cache'
                    resp.headers['Expires'] = '0'
                    return resp
        except Exception:
            pass

    # 2. Cloud RAM disk (/dev/shm) shared across Gunicorn workers
    if os.path.exists("/dev/shm/latest_cloud_frame.jpg"):
        try:
            with open("/dev/shm/latest_cloud_frame.jpg", "rb") as f:
                data = f.read()
                if len(data) > 100:
                    resp = Response(data, mimetype='image/jpeg')
                    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
                    resp.headers['Pragma'] = 'no-cache'
                    resp.headers['Expires'] = '0'
                    return resp
        except Exception:
            pass

    # 3. Disk file shared across all Gunicorn workers on Render
    live_path = os.path.join(IMAGE_FOLDER, "latest_live_frame.jpg")
    if os.path.exists(live_path):
        try:
            with open(live_path, "rb") as f:
                data = f.read()
                if len(data) > 100:
                    resp = Response(data, mimetype='image/jpeg')
                    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
                    resp.headers['Pragma'] = 'no-cache'
                    resp.headers['Expires'] = '0'
                    return resp
        except Exception:
            pass

    # 4. In-memory cloud frame fallback
    if latest_cloud_frame is not None and len(latest_cloud_frame) > 100:
        resp = Response(latest_cloud_frame, mimetype='image/jpeg')
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp

    # 5. Resilient fallback placeholder
    resp = Response(PLACEHOLDER_JPEG, mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route("/api/camera/snapshot")
def camera_snapshot():
    # 1. Try local port 8080 snapshot
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:8080/snapshot")
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            if resp.status == 200:
                r = Response(resp.read(), mimetype="image/jpeg")
                r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
                r.headers['Pragma'] = 'no-cache'
                r.headers['Expires'] = '0'
                return r
    except Exception:
        pass

    # 2. Try disk-persisted live frame
    live_path = os.path.join(IMAGE_FOLDER, "latest_live_frame.jpg")
    if os.path.exists(live_path):
        try:
            with open(live_path, "rb") as f:
                data = f.read()
            r = Response(data, mimetype="image/jpeg")
            r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
            r.headers['Pragma'] = 'no-cache'
            r.headers['Expires'] = '0'
            return r
        except Exception:
            pass

    # 3. Try in-memory buffer
    if latest_cloud_frame is not None:
        r = Response(latest_cloud_frame, mimetype="image/jpeg")
        r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        r.headers['Pragma'] = 'no-cache'
        r.headers['Expires'] = '0'
        return r

    # 4. Fallback to latest detection image
    try:
        latest = db.get_latest_detection()
        img_name = latest.get("image")
        if img_name:
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    data = f.read()
                r = Response(data, mimetype="image/jpeg")
                r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
                r.headers['Pragma'] = 'no-cache'
                r.headers['Expires'] = '0'
                return r
    except Exception:
        pass

    # 5. Resilient 1x1 black JPEG fallback
    r = Response(PLACEHOLDER_JPEG, mimetype="image/jpeg")
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    return r


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

    refresh_rover_connection_state()

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

        global _cached_analytics
        _cached_analytics = None

        # Clean up excess images if over 35 to prevent disk degradation
        try:
            protected_names = {"app_photo.png", "store_icon.png"}
            img_files = [
                os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
                and f not in protected_names
                and (f.startswith("2026") or f.startswith("leaf_"))
            ]
            if len(img_files) > 35:
                img_files.sort(key=os.path.getmtime)
                for old_f in img_files[:-35]:
                    try:
                        os.remove(old_f)
                    except Exception:
                        pass
        except Exception:
            pass

        rover_state["last_detection"] = disease
        rover_state["last_confidence"] = round(confidence_value, 1)

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
    methods=["GET", "POST"]
)
def rover_control():

    try:
        if request.method == "GET":
            data = request.args
        else:
            data = request.get_json(silent=True)
            if data is None:
                data = request.form

        command = str(
            data.get("command", "stop")
        ).lower().strip()

        valid_commands = [
            "forward",
            "backward",
            "left",
            "right",
            "stop",
            "start"
        ]

        if command not in valid_commands:
            return jsonify({
                "success": False,
                "message": "Invalid rover command",
                "command": command
            }), 400

        direction_names = {
            "forward": "Forward",
            "start": "Engine Armed",
            "backward": "Backward",
            "left": "Left",
            "right": "Right",
            "stop": "Stopped"
        }

        rover_state["direction"] = direction_names[command]
        rover_state["last_command"] = command
        rover_state["command_id"] += 1

        # Check for speed in payload
        req_speed = data.get("speed")
        if req_speed is not None:
            try:
                s = int(req_speed)
                if s > 0:
                    rover_state["speed"] = max(20, min(100, s))
            except Exception:
                pass

        if command in ("forward", "backward", "left", "right"):
            rover_state["engine_armed"] = True
            if rover_state.get("speed", 0) <= 0:
                rover_state["speed"] = rover_settings.get("speed", 60)
        elif command == "start":
            rover_state["engine_armed"] = True
            if rover_state.get("speed", 0) <= 0:
                rover_state["speed"] = rover_settings.get("speed", 60)
        elif command == "stop":
            rover_state["engine_armed"] = False
            rover_state["speed"] = 0

        print(
            "ROVER COMMAND:",
            command,
            "| Direction:",
            rover_state["direction"],
            "| Speed:",
            rover_state["speed"]
        )

        return jsonify({
            "success": True,
            "message": "Rover command received",
            "command": command,
            "command_id": rover_state["command_id"],
            "rover": rover_state
        })

    except Exception as e:
        print(
            "ROVER CONTROL ERROR:",
            e
        )
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ============================================================
# RASPBERRY PI COMMAND API
#
# Raspberry Pi will call this repeatedly.
# ============================================================

@app.route(
    "/api/rover/command",
    methods=["GET", "POST"]
)
def rover_command():
    if request.method == "POST":
        return rover_control()

    return jsonify({
        "success": True,
        "command": rover_state["last_command"],
        "command_id": rover_state["command_id"],
        "direction": rover_state["direction"],
        "speed": rover_state["speed"],
        "pump": rover_state["pump"],
        "mode": rover_state["mode"],
        "engine_armed": rover_state.get("engine_armed", False),
        "invert_right_motor": rover_settings.get("invert_right_motor", True),
        "invert_left_motor": rover_settings.get("invert_left_motor", False),
        "relay_active_low": rover_state.get("relay_active_low", True)
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
        data = request.get_json(silent=True) or request.form or {}
        now_epoch = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rover_state["last_seen_epoch"] = now_epoch
        rover_state["last_seen"] = now_str
        rover_state["connected"] = True
        rover_state["pi_online"] = True

        if data:
            if "model_online" in data:
                rover_state["model_online"] = bool(data.get("model_online"))
            else:
                rover_state["model_online"] = True

            if "device" in data:
                rover_state["device"] = str(data.get("device"))
            if "ai_status" in data:
                rover_state["ai_status"] = str(data.get("ai_status"))
            if "model_name" in data:
                rover_state["model_name"] = str(data.get("model_name"))
            if "fps" in data:
                try:
                    rover_state["fps"] = round(float(data.get("fps")), 1)
                except Exception:
                    pass
            if "last_detection" in data:
                rover_state["last_detection"] = str(data.get("last_detection"))
            if "last_confidence" in data:
                try:
                    rover_state["last_confidence"] = round(float(data.get("last_confidence")), 1)
                except Exception:
                    pass
            if "pi_ip" in data:
                rover_state["pi_ip"] = str(data.get("pi_ip"))
                rover_state["ip"] = str(data.get("pi_ip"))
            if "stream_url" in data:
                rover_state["stream_url"] = str(data.get("stream_url"))
            if "disease_name" in data:
                rover_state["disease_name"] = str(data.get("disease_name"))
            if "disease_chemical" in data:
                rover_state["disease_chemical"] = str(data.get("disease_chemical"))
            if "disease_fertilizer" in data:
                rover_state["disease_fertilizer"] = str(data.get("disease_fertilizer"))
            if "disease_treatment" in data:
                rover_state["disease_treatment"] = str(data.get("disease_treatment"))

        return jsonify({
            "success": True,
            "message": "Raspberry Pi & AI Model heartbeat acknowledged",
            "rover": rover_state
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

    refresh_rover_connection_state()

    return jsonify({

        "success": True,

        "rover":
            rover_state,

        "result":
            get_latest_result(),

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

        rover_state["command_id"] += 1


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
            "message": "Pump command received",
            "pump": state,
            "command": rover_state["last_command"],
            "command_id": rover_state["command_id"]
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route(
    "/api/rover/invert_relay",
    methods=["POST", "GET"]
)
def rover_invert_relay():
    try:
        rover_state["relay_active_low"] = not rover_state.get("relay_active_low", True)
        rover_state["command_id"] += 1
        return jsonify({
            "success": True,
            "relay_active_low": rover_state["relay_active_low"],
            "command_id": rover_state["command_id"],
            "message": f"Relay Mode: {'Active-Low (0V=ON)' if rover_state['relay_active_low'] else 'Active-High (3.3V=ON)'}"
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


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
        clean_res = DEFAULT_RESULT.copy()
        clean_res["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_json_file(RESULT_FILE, clean_res)
        try:
            db.clear_all_detections()
        except Exception:
            pass

        return jsonify({
            "success": True,
            "message": "Detection result and history cleared.",
            "result": clean_res
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ============================================================
# FAVICON
# ============================================================

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        STATIC_FOLDER,
        "favicon.ico",
        mimetype="image/x-icon"
    )


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
    import traceback
    return jsonify({
        "success": False,
        "error": str(error),
        "traceback": traceback.format_exc()
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
        debug=False,
        threaded=True
    )