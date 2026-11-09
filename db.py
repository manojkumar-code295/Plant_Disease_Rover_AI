"""
Plant Leaf Disease Detection & Smart Rover - SQLite Persistent Database.
Provides robust local and cloud data persistence:
  - Detection history with disease, confidence, image path, rover ID, spraying status
  - Spraying operation history (water only)
  - Telemetry & sensor logging
  - Thread-safe SQLite connection with WAL mode
"""

import os
import sqlite3
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "plants_data.db")
LEGACY_CSV = os.path.join(BASE_DIR, "detection_logs", "detection_history.csv")


def get_db_connection():
    """Returns an optimized SQLite connection with dictionary row access and WAL mode."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
    except Exception:
        pass
    return conn


def init_db():
    """Initializes all database tables and indexes."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id TEXT UNIQUE,
                date TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                chemical TEXT DEFAULT 'Not available',
                fertilizer TEXT DEFAULT 'Not available',
                treatment TEXT DEFAULT 'No treatment specified',
                image TEXT DEFAULT '',
                is_healthy INTEGER DEFAULT 0,
                rover_id TEXT DEFAULT 'ROVER-01',
                spraying_status TEXT DEFAULT 'NONE',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_date ON detections(date DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_disease ON detections(disease);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp DESC);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS spray_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                disease TEXT NOT NULL,
                duration REAL NOT NULL,
                confirmed_by_pi INTEGER DEFAULT 1,
                status TEXT DEFAULT 'COMPLETED',
                rover_id TEXT DEFAULT 'ROVER-01'
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                distance_cm REAL,
                battery_voltage REAL,
                battery_percent REAL,
                temperature_c REAL,
                humidity_pct REAL,
                soil_moisture_pct REAL,
                obstacle_alert INTEGER DEFAULT 0,
                rover_status TEXT DEFAULT 'CONNECTED'
            )
        """)
        conn.commit()

    migrate_legacy_csv()


def migrate_legacy_csv():
    """Seeds database from legacy detection_history.csv if empty."""
    if not os.path.exists(LEGACY_CSV):
        return

    try:
        with get_db_connection() as conn:
            cnt = conn.execute("SELECT count(*) as c FROM detections").fetchone()["c"]
            if cnt > 0:
                return

            import csv
            with open(LEGACY_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dis = row.get("Disease", "Unknown")
                    conf = 0.0
                    try:
                        conf = float(row.get("Confidence", "0").replace("%", ""))
                    except Exception:
                        pass
                    ts = row.get("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    dt = ts.split(" ")[0] if " " in ts else ts
                    img = row.get("Image_File", "")
                    healthy = 1 if "healthy" in dis.lower() else 0
                    sprayed = row.get("Spraying_Action", "NONE")
                    conn.execute("""
                        INSERT OR IGNORE INTO detections 
                        (result_id, date, timestamp, disease, confidence, image, is_healthy, spraying_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (f"seed_{os.urandom(4).hex()}", dt, ts, dis, conf, img, healthy, sprayed))
            conn.commit()
    except Exception as e:
        print(f"[DB] Migration note: {e}")


def save_detection(data):
    """Saves a plant disease detection record reliably to SQLite."""
    if not isinstance(data, dict):
        return None

    disease = str(data.get("disease") or data.get("predicted_class") or "Unknown").strip()
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    timestamp = data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = data.get("date")
    if not date_str:
        try:
            date_str = timestamp.split(" ")[0]
        except Exception:
            date_str = datetime.now().strftime("%Y-%m-%d")

    result_id = str(data.get("result_id") or f"det_{int(datetime.now().timestamp())}_{os.urandom(3).hex()}")
    chemical = str(data.get("chemical", "Not available"))
    fertilizer = str(data.get("fertilizer", "Not available"))
    treatment = str(data.get("treatment") or data.get("treatment_notes") or "Point camera at plant leaf to inspect.")
    image = str(data.get("image") or data.get("image_filename") or "")
    rover_id = str(data.get("rover_id", "ROVER-01"))
    spraying_status = str(data.get("spraying_status") or data.get("spraying_action") or "NONE")

    is_healthy = 1 if ("healthy" in disease.lower() and "unhealthy" not in disease.lower()) else 0

    record = {
        "result_id": result_id,
        "date": date_str,
        "timestamp": timestamp,
        "disease": disease,
        "confidence": round(confidence, 2),
        "chemical": chemical,
        "fertilizer": fertilizer,
        "treatment": treatment,
        "image": image,
        "is_healthy": is_healthy,
        "rover_id": rover_id,
        "spraying_status": spraying_status
    }

    try:
        with get_db_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO detections 
                (result_id, date, timestamp, disease, confidence, chemical, fertilizer, treatment, image, is_healthy, rover_id, spraying_status)
                VALUES (:result_id, :date, :timestamp, :disease, :confidence, :chemical, :fertilizer, :treatment, :image, :is_healthy, :rover_id, :spraying_status)
            """, record)
            conn.commit()
        return record
    except Exception as e:
        print(f"[DB] Save detection error: {e}")
        return record


def get_latest_detection():
    """Returns the most recent single detection result."""
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
            if row:
                return dict(row)
    except Exception:
        pass

    return {
        "result_id": "",
        "image": "",
        "disease": "No leaf scanned yet",
        "chemical": "None required",
        "fertilizer": "None required",
        "treatment": "Waiting for live camera frame or image upload.",
        "confidence": 0.0,
        "timestamp": "",
        "date": "",
        "is_healthy": 1,
        "rover_id": "ROVER-01",
        "spraying_status": "NONE"
    }


def get_recent_history(limit=50, category=None):
    """Returns detection records as a list of dicts with optional category filter."""
    try:
        with get_db_connection() as conn:
            if category and category.lower() != "all":
                rows = conn.execute(
                    "SELECT * FROM detections WHERE LOWER(disease) LIKE ? ORDER BY id DESC LIMIT ?",
                    (f"%{category.lower()}%", limit)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB] Get history error: {e}")
        return []


def get_gallery_images():
    """Returns all detected images organized into the 4 target categories."""
    categories = {
        "Healthy": [],
        "Cercospora": [],
        "Anthracnose": [],
        "Bacterial_Spot": []
    }
    try:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM detections WHERE image != '' ORDER BY id DESC").fetchall()
            for r in rows:
                rec = dict(r)
                dis = rec.get("disease", "")
                cat_key = "Healthy"
                if "cercospora" in dis.lower():
                    cat_key = "Cercospora"
                elif "anthracnose" in dis.lower():
                    cat_key = "Anthracnose"
                elif "bacterial" in dis.lower():
                    cat_key = "Bacterial_Spot"
                elif "healthy" in dis.lower():
                    cat_key = "Healthy"
                
                if cat_key in categories and len(categories[cat_key]) < 24:
                    categories[cat_key].append(rec)
    except Exception as e:
        print(f"[DB] Get gallery error: {e}")
    return categories


def save_spray_event(disease, duration, confirmed=True, rover_id="ROVER-01"):
    """Records an executed water spray event."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "CONFIRMED_SPRAYED" if confirmed else "UNCONFIRMED"
    try:
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO spray_history (timestamp, disease, duration, confirmed_by_pi, status, rover_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ts, disease, duration, 1 if confirmed else 0, status, rover_id))
            conn.commit()
    except Exception as e:
        print(f"[DB] Save spray error: {e}")


def get_spray_history(limit=20):
    """Returns the latest confirmed spraying events."""
    try:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM spray_history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB] Get spray history error: {e}")
        return []


def get_stats():
    """Computes summary statistics for dashboard counters."""
    stats = {
        "total": 0,
        "healthy": 0,
        "diseased": 0,
        "healthy_percent": 100,
        "diseased_percent": 0,
        "total_sprays": 0
    }
    try:
        with get_db_connection() as conn:
            tot = conn.execute("SELECT count(*) as c FROM detections").fetchone()["c"]
            hlth = conn.execute("SELECT count(*) as c FROM detections WHERE is_healthy = 1").fetchone()["c"]
            sprays = conn.execute("SELECT count(*) as c FROM spray_history").fetchone()["c"]
            stats["total"] = tot
            stats["healthy"] = hlth
            stats["diseased"] = max(0, tot - hlth)
            stats["total_sprays"] = sprays
            if tot > 0:
                stats["healthy_percent"] = round((hlth / tot) * 100, 1)
                stats["diseased_percent"] = round(((tot - hlth) / tot) * 100, 1)
    except Exception as e:
        print(f"[DB] Get stats error: {e}")
    return stats


# Auto-initialize database on import
init_db()
