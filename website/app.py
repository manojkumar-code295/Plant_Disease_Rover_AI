from flask import Flask, render_template
import json
import os
import sqlite3

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


# =========================================================
# READ LATEST AI RESULT
# =========================================================

def get_latest_result():

    if not os.path.exists(RESULT_FILE):

        return {
            "image": "No image",
            "disease": "No detection yet",
            "chemical": "Waiting for AI result",
            "fertilizer": "Waiting for AI result",
            "treatment": "Run the AI prediction first."
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
            "image": "Error",
            "disease": "Unable to read result",
            "chemical": "-",
            "fertilizer": "-",
            "treatment": "-"
        }


# =========================================================
# READ PREDICTION HISTORY
# =========================================================

def get_prediction_history():

    if not os.path.exists(DATABASE_FILE):

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


    if not os.path.exists(DATABASE_FILE):

        return analytics


    try:

        connection = sqlite3.connect(
            DATABASE_FILE
        )

        cursor = connection.cursor()


        # =================================================
        # TOTAL PREDICTIONS
        # =================================================

        cursor.execute("""
            SELECT COUNT(*)
            FROM predictions
        """)

        analytics["total"] = cursor.fetchone()[0]


        # =================================================
        # HEALTHY LEAVES
        # =================================================

        cursor.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE disease = 'Healthy Leaf'
        """)

        analytics["healthy"] = cursor.fetchone()[0]


        # =================================================
        # DISEASED LEAVES
        # =================================================

        analytics["diseased"] = (
            analytics["total"]
            -
            analytics["healthy"]
        )


        # =================================================
        # BACTERIAL SPOT
        # =================================================

        cursor.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE disease = 'Bacterial Spot'
        """)

        analytics["bacterial_spot"] = (
            cursor.fetchone()[0]
        )


        # =================================================
        # CERCOSPORA
        # =================================================

        cursor.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE disease = 'Cercospora Leaf Spot'
        """)

        analytics["cercospora"] = (
            cursor.fetchone()[0]
        )


        # =================================================
        # CURL VIRUS
        # =================================================

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

    statistics = get_analytics()

    return render_template(
        "dashboard.html",
        detection=result,
        statistics=statistics
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
# DATASET
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
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )