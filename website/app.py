from flask import Flask, render_template, request, jsonify
import json
import os
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULT_FILE = os.path.join(
    BASE_DIR,
    "result.json"
)

DATABASE_FILE = os.path.join(
    BASE_DIR,
    "rover.db"
)

IMAGE_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "images"
)


# =========================================================
# CREATE REQUIRED FOLDERS
# =========================================================

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def initialize_database():

    try:

        connection = sqlite3.connect(
            DATABASE_FILE
        )

        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                image TEXT,

                disease TEXT,

                chemical TEXT,

                fertilizer TEXT,

                treatment TEXT,

                date TEXT,

                time TEXT
            )
        """)

        connection.commit()

        connection.close()

        print(
            "Database initialized successfully."
        )

    except Exception as e:

        print(
            "Database initialization error:",
            e
        )


initialize_database()


# =========================================================
# READ LATEST AI RESULT
# =========================================================

def get_latest_result():

    if not os.path.exists(
        RESULT_FILE
    ):

        return {

            "image": "",

            "disease":
                "No detection yet",

            "chemical":
                "Waiting for AI result",

            "fertilizer":
                "Waiting for AI result",

            "treatment":
                "Run the AI prediction first."
        }

    try:

        with open(
            RESULT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as e:

        print(
            "Error reading result.json:",
            e
        )

        return {

            "image": "",

            "disease":
                "Unable to read result",

            "chemical":
                "-",

            "fertilizer":
                "-",

            "treatment":
                "-"
        }


# =========================================================
# READ PREDICTION HISTORY
# =========================================================

def get_prediction_history():

    if not os.path.exists(
        DATABASE_FILE
    ):

        return []

    try:

        connection = sqlite3.connect(
            DATABASE_FILE
        )

        connection.row_factory = sqlite3.Row

        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                id,
                image,
                disease,
                chemical,
                fertilizer,
                treatment,
                date,
                time
            FROM predictions
            ORDER BY id DESC
        """)

        rows = cursor.fetchall()

        connection.close()

        return rows

    except Exception as e:

        print(
            "Error reading database:",
            e
        )

        return []


# =========================================================
# GET ANALYTICS DATA
# =========================================================

def get_analytics():

    analytics = {

        "total": 0,

        "healthy": 0,

        "diseased": 0,

        "bacterial_spot": 0,

        "cercospora": 0,

        "curl_virus": 0
    }

    if not os.path.exists(
        DATABASE_FILE
    ):

        return analytics

    try:

        connection = sqlite3.connect(
            DATABASE_FILE
        )

        cursor = connection.cursor()


        # ---------------------------------------------
        # TOTAL PREDICTIONS
        # ---------------------------------------------

        cursor.execute("""
            SELECT COUNT(*)
            FROM predictions
        """)

        analytics["total"] = (
            cursor.fetchone()[0]
        )


        # ---------------------------------------------
        # HEALTHY LEAVES
        # ---------------------------------------------

        cursor.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE disease = 'Healthy Leaf'
        """)

        analytics["healthy"] = (
            cursor.fetchone()[0]
        )


        # ---------------------------------------------
        # DISEASED
        # ---------------------------------------------

        analytics["diseased"] = (

            analytics["total"]
            -
            analytics["healthy"]
        )


        # ---------------------------------------------
        # BACTERIAL SPOT
        # ---------------------------------------------

        cursor.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE disease = 'Bacterial Spot'
        """)

        analytics["bacterial_spot"] = (
            cursor.fetchone()[0]
        )


        # ---------------------------------------------
        # CERCOSPORA
        # ---------------------------------------------

        cursor.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE disease = 'Cercospora Leaf Spot'
        """)

        analytics["cercospora"] = (
            cursor.fetchone()[0]
        )


        # ---------------------------------------------
        # CURL VIRUS
        # ---------------------------------------------

        cursor.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE disease = 'Curl Virus'
        """)

        analytics["curl_virus"] = (
            cursor.fetchone()[0]
        )


        connection.close()

        return analytics

    except Exception as e:

        print(
            "Error calculating analytics:",
            e
        )

        return analytics


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/")
def dashboard():

    result = get_latest_result()

    return render_template(

        "dashboard.html",

        detection=result
    )


# =========================================================
# ROVER CONTROL
# =========================================================

@app.route("/rover")
def rover():

    return render_template(
        "rover.html"
    )


# =========================================================
# DATASET / PREDICTION HISTORY
# =========================================================

@app.route("/dataset")
def dataset():

    history = get_prediction_history()

    return render_template(

        "dataset.html",

        history=history
    )


# =========================================================
# ANALYTICS
# =========================================================

@app.route("/analytics")
def analytics():

    data = get_analytics()

    return render_template(

        "analytics.html",

        data=data
    )


# =========================================================
# SETTINGS
# =========================================================

@app.route("/settings")
def settings():

    return render_template(
        "settings.html"
    )


# =========================================================
# API HEALTH CHECK
# =========================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "online",

        "service":
            "Precision Agriculture Smart Rover",

        "api":
            "working"
    })


# =========================================================
# API - RECEIVE AI RESULT
# =========================================================

@app.route(
    "/api/update-result",
    methods=["POST"]
)
def update_result():

    try:

        print()
        print(
            "========================================"
        )

        print(
            "      NEW AI RESULT RECEIVED"
        )

        print(
            "========================================"
        )


        # =================================================
        # READ FORM DATA
        # =================================================

        disease = request.form.get(
            "disease",
            "Unknown"
        )

        chemical = request.form.get(
            "chemical",
            "-"
        )

        fertilizer = request.form.get(
            "fertilizer",
            "-"
        )

        treatment = request.form.get(
            "treatment",
            "-"
        )


        # =================================================
        # READ IMAGE
        # =================================================

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


        # =================================================
        # SECURE IMAGE NAME
        # =================================================

        filename = secure_filename(
            image.filename
        )

        if not filename:

            return jsonify({

                "success":
                    False,

                "message":
                    "Invalid image filename"
            }), 400


        # =================================================
        # SAVE IMAGE
        # =================================================

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


        # =================================================
        # GET CURRENT DATE AND TIME
        # =================================================

        now = datetime.now()

        current_date = (
            now.strftime("%Y-%m-%d")
        )

        current_time = (
            now.strftime("%H:%M:%S")
        )


        # =================================================
        # CREATE RESULT
        # =================================================

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
                treatment
        }


        # =================================================
        # SAVE result.json
        # =================================================

        with open(

            RESULT_FILE,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                result,

                file,

                indent=4
            )


        print(
            "Result saved to result.json."
        )


        # =================================================
        # SAVE PREDICTION TO DATABASE
        # =================================================

        try:

            connection = sqlite3.connect(
                DATABASE_FILE
            )

            cursor = connection.cursor()

            cursor.execute("""
                INSERT INTO predictions (

                    image,
                    disease,
                    chemical,
                    fertilizer,
                    treatment,
                    date,
                    time

                )

                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (

                filename,

                disease,

                chemical,

                fertilizer,

                treatment,

                current_date,

                current_time
            ))

            connection.commit()

            connection.close()

            print(
                "Prediction added to database."
            )

        except Exception as e:

            print(
                "Database save error:",
                e
            )


        # =================================================
        # SUCCESS RESPONSE
        # =================================================

        print()
        print(
            "========================================"
        )

        print(
            "      AI RESULT UPDATED SUCCESSFULLY"
        )

        print(
            "========================================"
        )

        print(
            "Disease:",
            disease
        )

        print(
            "Image:",
            filename
        )


        return jsonify({

            "success":
                True,

            "message":
                "AI result and image updated successfully",

            "result":
                result

        }), 200


    except Exception as e:

        print()
        print(
            "========================================"
        )

        print(
            "           API ERROR"
        )

        print(
            "========================================"
        )

        print(e)


        return jsonify({

            "success":
                False,

            "message":
                str(e)

        }), 500


# =========================================================
# OPTIONAL - CLEAR CURRENT RESULT
# =========================================================

@app.route(
    "/api/clear-result",
    methods=["POST"]
)
def clear_result():

    try:

        result = {

            "image": "",

            "disease":
                "No detection yet",

            "chemical":
                "Waiting for AI result",

            "fertilizer":
                "Waiting for AI result",

            "treatment":
                "Run the AI prediction first."
        }

        with open(

            RESULT_FILE,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                result,

                file,

                indent=4
            )


        return jsonify({

            "success":
                True,

            "message":
                "Current result cleared"
        })


    except Exception as e:

        return jsonify({

            "success":
                False,

            "message":
                str(e)

        }), 500


# =========================================================
# ERROR HANDLER - PAGE NOT FOUND
# =========================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({

        "success":
            False,

        "message":
            "Requested URL was not found on the server."

    }), 404


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    port = int(

        os.environ.get(

            "PORT",

            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False
    )