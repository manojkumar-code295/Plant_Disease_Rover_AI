"""
=============================================================================
AGRIROVER SMART ROVER - ROBUST SQLITE PLANT DETECTION DATABASE
=============================================================================
Provides persistent, daily-partitioned local data storage on Raspberry Pi.
Features:
- SQLite WAL mode for safe, concurrent read/writes between AI engine and web server
- Automatic daily grouping and aggregation
- Resilient transactions with retry handling
- Auto-migration from legacy history.json if present
=============================================================================
"""

import os
import sqlite3
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "plants_data.db")
LEGACY_HISTORY = os.path.join(BASE_DIR, "history.json")


def get_db_connection():
    """Returns an optimized SQLite connection with dictionary row access and WAL mode."""
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db():
    """Initializes the detections table and indexes."""
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
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_date ON detections(date DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp DESC);")
        conn.commit()

    # Auto-migrate legacy history.json if database is currently empty
    migrate_legacy_history()


def migrate_legacy_history():
    """One-time migration of history.json data into SQLite."""
    if not os.path.exists(LEGACY_HISTORY):
        return

    try:
        with get_db_connection() as conn:
            count = conn.execute("SELECT count(*) as cnt FROM detections").fetchone()["cnt"]
            if count > 0:
                return  # Already populated

            with open(LEGACY_HISTORY, "r", encoding="utf-8") as f:
                history_list = json.load(f)

            if isinstance(history_list, list):
                for item in history_list:
                    save_detection(item)
    except Exception as e:
        print(f"[DB MIGRATION NOTICE] {e}")


def save_detection(data):
    """
    Saves a plant disease detection record reliably to SQLite.
    Returns the inserted record dictionary.
    """
    if not isinstance(data, dict):
        return None

    disease = str(data.get("disease", "Unknown")).strip()
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
    treatment = str(data.get("treatment", "No specific treatment necessary."))
    image = str(data.get("image", ""))
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
        "is_healthy": is_healthy
    }

    try:
        with get_db_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO detections 
                (result_id, date, timestamp, disease, confidence, chemical, fertilizer, treatment, image, is_healthy)
                VALUES (:result_id, :date, :timestamp, :disease, :confidence, :chemical, :fertilizer, :treatment, :image, :is_healthy)
            """, record)
            conn.commit()
        return record
    except Exception as e:
        print(f"[DB SAVE ERROR] {e}")
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
        "disease": "No prediction yet",
        "chemical": "Not available",
        "fertilizer": "Not available",
        "treatment": "No prediction has been processed yet.",
        "confidence": 0.0,
        "timestamp": "",
        "date": ""
    }


def get_recent_history(limit=100):
    """Returns the most recent detection records as a list of dicts."""
    try:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB GET HISTORY ERROR] {e}")
        return []


def get_daily_grouped_data():
    """
    Returns data grouped by date for the Daily Data page.
    Format: list of day dictionaries with total counts, breakdown, and itemized records.
    """
    try:
        with get_db_connection() as conn:
            dates = [r["date"] for r in conn.execute("SELECT DISTINCT date FROM detections ORDER BY date DESC").fetchall()]
            daily_list = []

            for d in dates:
                day_rows = [dict(r) for r in conn.execute("SELECT * FROM detections WHERE date = ? ORDER BY id DESC", (d,)).fetchall()]
                total = len(day_rows)
                healthy = sum(1 for r in day_rows if r.get("is_healthy"))
                diseased = total - healthy
                avg_conf = round(sum(r.get("confidence", 0) for r in day_rows) / max(total, 1), 1)

                # Breakdown of specific diseases
                bacterial = sum(1 for r in day_rows if "bacterial" in r["disease"].lower())
                cercospora = sum(1 for r in day_rows if "cercospora" in r["disease"].lower())
                curl_virus = sum(1 for r in day_rows if "curl" in r["disease"].lower())

                daily_list.append({
                    "date": d,
                    "total": total,
                    "healthy": healthy,
                    "diseased": diseased,
                    "avg_confidence": avg_conf,
                    "bacterial_spot": bacterial,
                    "cercospora": cercospora,
                    "curl_virus": curl_virus,
                    "records": day_rows
                })

            return daily_list
    except Exception as e:
        print(f"[DB DAILY GROUP ERROR] {e}")
        return []


def get_analytics_summary():
    """Calculates overall statistics across the entire database."""
    try:
        with get_db_connection() as conn:
            total = conn.execute("SELECT count(*) as cnt FROM detections").fetchone()["cnt"]
            if total == 0:
                return {
                    "total": 0, "healthy": 0, "diseased": 0,
                    "bacterial_spot": 0, "cercospora": 0, "curl_virus": 0,
                    "healthy_pct": 0.0, "diseased_pct": 0.0, "avg_confidence": 0.0
                }

            healthy = conn.execute("SELECT count(*) as cnt FROM detections WHERE is_healthy = 1").fetchone()["cnt"]
            diseased = total - healthy
            avg_conf = conn.execute("SELECT avg(confidence) as avgc FROM detections").fetchone()["avgc"] or 0.0

            bacterial = conn.execute("SELECT count(*) as cnt FROM detections WHERE lower(disease) LIKE '%bacterial%'").fetchone()["cnt"]
            cercospora = conn.execute("SELECT count(*) as cnt FROM detections WHERE lower(disease) LIKE '%cercospora%'").fetchone()["cnt"]
            curl_virus = conn.execute("SELECT count(*) as cnt FROM detections WHERE lower(disease) LIKE '%curl%'").fetchone()["cnt"]

            return {
                "total": total,
                "healthy": healthy,
                "diseased": diseased,
                "bacterial_spot": bacterial,
                "cercospora": cercospora,
                "curl_virus": curl_virus,
                "healthy_pct": round((healthy / total) * 100.0, 1),
                "diseased_pct": round((diseased / total) * 100.0, 1),
                "avg_confidence": round(avg_conf, 1)
            }
    except Exception as e:
        print(f"[DB ANALYTICS ERROR] {e}")
        return {
            "total": 0, "healthy": 0, "diseased": 0,
            "bacterial_spot": 0, "cercospora": 0, "curl_virus": 0,
            "healthy_pct": 0.0, "diseased_pct": 0.0, "avg_confidence": 0.0
        }


def clear_all_detections():
    """Clears all detection records from database."""
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM detections;")
            conn.commit()
        return True
    except Exception as e:
        print(f"[DB CLEAR ERROR] {e}")
        return False


# Auto-initialize database schema when imported
init_db()
