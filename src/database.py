"""
Database persistence layer using SQLite for Animal Digital Twins.
Thread-safe and proper connection lifecycle management for Windows compatibility.
"""
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

from src.config import DATABASE_PATH, DATA_DIR


def _create_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create a raw SQLite database connection with row factory enabled."""
    if db_path is None:
        db_path = DATABASE_PATH
    
    os.makedirs(db_path.parent, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db_connection(db_path: Optional[Path] = None):
    """Context manager to ensure connections are always closed promptly."""
    conn = _create_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables and seed default profiles if needed."""
    with get_db_connection(db_path) as conn:
        with conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS animals (
                animal_id TEXT PRIMARY KEY,
                species TEXT NOT NULL DEFAULT 'Cattle',
                name TEXT,
                breed TEXT,
                birth_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                animal_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                observation_type TEXT NOT NULL DEFAULT 'image',
                cv_prediction TEXT NOT NULL,
                cv_confidence REAL NOT NULL,
                model_used TEXT,
                notes TEXT,
                FOREIGN KEY (animal_id) REFERENCES animals(animal_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                animal_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                body_temperature_c REAL,
                activity_level TEXT,
                feed_intake TEXT,
                ambient_temperature_c REAL,
                humidity_percent REAL,
                data_source TEXT NOT NULL DEFAULT 'Manual Entry',
                is_simulated INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (animal_id) REFERENCES animals(animal_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                animal_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                risk_status TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                FOREIGN KEY (animal_id) REFERENCES animals(animal_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS dmi_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                animal_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                predicted_dmi_kg_day REAL NOT NULL,
                model_name TEXT NOT NULL,
                input_features_json TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY (animal_id) REFERENCES animals(animal_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feed_inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feed_name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                available_stock_kg REAL NOT NULL CHECK (available_stock_kg >= 0),
                daily_usage_kg REAL NOT NULL CHECK (daily_usage_kg >= 0),
                unit_cost REAL NOT NULL CHECK (unit_cost >= 0),
                supplier_name TEXT NOT NULL DEFAULT '',
                supplier_lead_time_days REAL NOT NULL CHECK (supplier_lead_time_days >= 0),
                reorder_threshold_kg REAL NOT NULL CHECK (reorder_threshold_kg >= 0),
                last_updated TEXT NOT NULL
            );
            """)

        cursor = conn.execute("SELECT COUNT(*) FROM animals")
        count = cursor.fetchone()[0]
        if count == 0:
            seed_default_profiles(conn)


def seed_default_profiles(conn: sqlite3.Connection) -> None:
    """
    Seed initial prototype animals into the database.
    Animal C001 corresponds directly to the prototype in the user's research notebook.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Seed C001 (Prototype animal from notebookef7b497bba (1).ipynb)
    conn.execute(
        "INSERT INTO animals (animal_id, species, name, breed, birth_date, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("C001", "Cattle", "Daisy", "Holstein Friesian", "2023-04-12", "2026-09-16 10:00:00", now_str)
    )

    conn.execute(
        "INSERT INTO observations (animal_id, timestamp, observation_type, cv_prediction, cv_confidence, model_used, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("C001", "2026-09-16 14:15:20", "image", "lumpy", 99.99, "DenseNet-121", "Initial skin lesions observed")
    )
    conn.execute(
        "INSERT INTO observations (animal_id, timestamp, observation_type, cv_prediction, cv_confidence, model_used, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("C001", "2026-09-16 18:30:10", "image", "healthy", 100.0, "DenseNet-121", "Follow-up screening inspection")
    )

    conn.execute(
        "INSERT INTO sensor_readings (animal_id, timestamp, body_temperature_c, activity_level, feed_intake, "
        "ambient_temperature_c, humidity_percent, data_source, is_simulated) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("C001", "2026-09-16 18:35:00", 39.2, "Low", "Reduced", 32.0, 78.0, "Simulated (Collar Sensor)", 1)
    )

    reasons_c001 = [
        "Previous observation indicated disease.",
        "Body temperature is elevated in this prototype scenario.",
        "Low activity level was recorded.",
        "Reduced feed intake was recorded.",
        "High ambient temperature provides additional environmental context.",
        "High humidity provides additional environmental context."
    ]
    conn.execute(
        "INSERT INTO risk_assessments (animal_id, timestamp, risk_score, risk_status, reasons_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("C001", "2026-09-16 18:35:05", 6, "High", json.dumps(reasons_c001))
    )

    # Seed C002 (Healthy baseline cow)
    conn.execute(
        "INSERT INTO animals (animal_id, species, name, breed, birth_date, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("C002", "Cattle", "Bella", "Jersey", "2024-01-15", now_str, now_str)
    )
    conn.execute(
        "INSERT INTO observations (animal_id, timestamp, observation_type, cv_prediction, cv_confidence, model_used, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("C002", now_str, "image", "healthy", 99.87, "DenseNet-121", "Routine herd check")
    )
    conn.execute(
        "INSERT INTO sensor_readings (animal_id, timestamp, body_temperature_c, activity_level, feed_intake, "
        "ambient_temperature_c, humidity_percent, data_source, is_simulated) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("C002", now_str, 38.5, "Normal", "Normal", 24.0, 52.0, "Simulated (Collar Sensor)", 1)
    )
    reasons_c002 = ["No disease condition is currently present in the recorded history."]
    conn.execute(
        "INSERT INTO risk_assessments (animal_id, timestamp, risk_score, risk_status, reasons_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("C002", now_str, 0, "Low", json.dumps(reasons_c002))
    )

    # Seed C003 (Foot-and-Mouth suspect)
    conn.execute(
        "INSERT INTO animals (animal_id, species, name, breed, birth_date, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("C003", "Cattle", "Luna", "Ayrshire", "2023-11-20", now_str, now_str)
    )
    conn.execute(
        "INSERT INTO observations (animal_id, timestamp, observation_type, cv_prediction, cv_confidence, model_used, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("C003", now_str, "image", "foot-and-mouth", 99.98, "DenseNet-121", "Oral lesions and drooling noted")
    )
    conn.execute(
        "INSERT INTO sensor_readings (animal_id, timestamp, body_temperature_c, activity_level, feed_intake, "
        "ambient_temperature_c, humidity_percent, data_source, is_simulated) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("C003", now_str, 40.1, "Low", "Reduced", 28.0, 65.0, "Manual Farm Inspection", 0)
    )
    reasons_c003 = [
        "Current CV observation indicates foot-and-mouth.",
        "Body temperature is elevated in this prototype scenario.",
        "Low activity level was recorded.",
        "Reduced feed intake was recorded."
    ]
    conn.execute(
        "INSERT INTO risk_assessments (animal_id, timestamp, risk_score, risk_status, reasons_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("C003", now_str, 7, "High", json.dumps(reasons_c003))
    )
    conn.commit()


# =========================================================================
# Animal CRUD Operations
# =========================================================================

def create_animal(
    animal_id: str,
    species: str = "Cattle",
    name: Optional[str] = None,
    breed: Optional[str] = None,
    birth_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> bool:
    """Create a new animal profile in the database."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection(db_path) as conn:
        try:
            with conn:
                conn.execute(
                    "INSERT INTO animals (animal_id, species, name, breed, birth_date, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (animal_id.strip(), species.strip(), name.strip() if name else None,
                     breed.strip() if breed else None, birth_date, now_str, now_str)
                )
            return True
        except sqlite3.IntegrityError:
            raise ValueError(f"Animal ID '{animal_id}' already exists.")


def get_animal(animal_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve an animal profile by ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM animals WHERE animal_id = ?", (animal_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_animals(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List all animal profiles ordered by ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM animals ORDER BY animal_id ASC")
        return [dict(r) for r in cursor.fetchall()]


def delete_animal(animal_id: str, db_path: Optional[Path] = None) -> bool:
    """Delete an animal and all associated records (cascaded)."""
    with get_db_connection(db_path) as conn:
        with conn:
            cursor = conn.execute("DELETE FROM animals WHERE animal_id = ?", (animal_id,))
            return cursor.rowcount > 0


# =========================================================================
# Observations & Sensor Readings Operations
# =========================================================================

def add_observation(
    animal_id: str,
    cv_prediction: str,
    cv_confidence: float,
    model_used: str = "DenseNet-121",
    observation_type: str = "image",
    timestamp: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Insert a new health observation for the animal."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with get_db_connection(db_path) as conn:
        with conn:
            cursor = conn.execute(
                "INSERT INTO observations (animal_id, timestamp, observation_type, cv_prediction, cv_confidence, model_used, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (animal_id, timestamp, observation_type, cv_prediction, float(cv_confidence), model_used, notes)
            )
            conn.execute("UPDATE animals SET updated_at = ? WHERE animal_id = ?", (timestamp, animal_id))
            return cursor.lastrowid


def get_observations(animal_id: str, limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve health observations for an animal ordered chronologically."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM observations WHERE animal_id = ? ORDER BY timestamp ASC, id ASC LIMIT ?",
            (animal_id, limit)
        )
        return [dict(r) for r in cursor.fetchall()]


def get_latest_observation(animal_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Get the most recent observation for an animal."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM observations WHERE animal_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
            (animal_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def add_sensor_reading(
    animal_id: str,
    body_temperature_c: Optional[float],
    activity_level: Optional[str],
    feed_intake: Optional[str],
    ambient_temperature_c: Optional[float],
    humidity_percent: Optional[float],
    data_source: str = "Manual Entry",
    is_simulated: bool = False,
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Insert a new sensor / environmental reading for the animal."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with get_db_connection(db_path) as conn:
        with conn:
            cursor = conn.execute(
                "INSERT INTO sensor_readings (animal_id, timestamp, body_temperature_c, activity_level, feed_intake, "
                "ambient_temperature_c, humidity_percent, data_source, is_simulated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (animal_id, timestamp, body_temperature_c, activity_level, feed_intake,
                 ambient_temperature_c, humidity_percent, data_source, 1 if is_simulated else 0)
            )
            conn.execute("UPDATE animals SET updated_at = ? WHERE animal_id = ?", (timestamp, animal_id))
            return cursor.lastrowid


def get_sensor_readings(animal_id: str, limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get sensor readings for an animal ordered chronologically."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM sensor_readings WHERE animal_id = ? ORDER BY timestamp ASC, id ASC LIMIT ?",
            (animal_id, limit)
        )
        return [dict(r) for r in cursor.fetchall()]


def get_latest_sensor_reading(animal_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Get the most recent sensor reading for an animal."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM sensor_readings WHERE animal_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
            (animal_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def save_risk_assessment(
    animal_id: str,
    risk_score: int,
    risk_status: str,
    reasons: List[str],
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Save a computed risk assessment."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    reasons_json = json.dumps(reasons)
    with get_db_connection(db_path) as conn:
        with conn:
            cursor = conn.execute(
                "INSERT INTO risk_assessments (animal_id, timestamp, risk_score, risk_status, reasons_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (animal_id, timestamp, risk_score, risk_status, reasons_json)
            )
            return cursor.lastrowid


def get_latest_risk_assessment(animal_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Get the most recent risk assessment for an animal."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM risk_assessments WHERE animal_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
            (animal_id,)
        )
        row = cursor.fetchone()
        if row:
            res = dict(row)
            res["reasons"] = json.loads(res["reasons_json"])
            return res
        return None


# =========================================================================
# DMI (Dry Matter Intake) Operations
# =========================================================================

def add_dmi_prediction(
    animal_id: str,
    predicted_dmi_kg_day: float,
    model_name: str,
    input_features: Dict[str, Any],
    notes: Optional[str] = None,
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Store an experimental daily Dry Matter Intake prediction."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    features_json = json.dumps(input_features)
    with get_db_connection(db_path) as conn:
        with conn:
            cursor = conn.execute(
                "INSERT INTO dmi_predictions (animal_id, timestamp, predicted_dmi_kg_day, model_name, input_features_json, notes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (animal_id, timestamp, float(predicted_dmi_kg_day), model_name, features_json, notes)
            )
            conn.execute("UPDATE animals SET updated_at = ? WHERE animal_id = ?", (timestamp, animal_id))
            return cursor.lastrowid


def get_dmi_predictions(animal_id: str, limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve historical DMI predictions for an animal in chronological order."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM dmi_predictions WHERE animal_id = ? ORDER BY timestamp ASC, id ASC LIMIT ?",
            (animal_id, limit)
        )
        results = []
        for r in cursor.fetchall():
            d = dict(r)
            d["input_features"] = json.loads(d["input_features_json"])
            results.append(d)
        return results


def get_latest_dmi_prediction(animal_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Get the most recent DMI prediction for an animal."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM dmi_predictions WHERE animal_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
            (animal_id,)
        )
        row = cursor.fetchone()
        if row:
            d = dict(row)
            d["input_features"] = json.loads(d["input_features_json"])
            return d
        return None


# =========================================================================
# Feed Inventory Operations
# =========================================================================

def upsert_feed_inventory_item(
    feed_name: str,
    available_stock_kg: float,
    daily_usage_kg: float,
    unit_cost: float,
    supplier_name: str,
    supplier_lead_time_days: float,
    reorder_threshold_kg: float,
    db_path: Optional[Path] = None,
) -> int:
    """Insert a feed item or update its current inventory values by name."""
    from src.feed_inventory import validate_feed_inventory_item

    item = validate_feed_inventory_item({
        "feed_name": feed_name,
        "available_stock_kg": available_stock_kg,
        "daily_usage_kg": daily_usage_kg,
        "unit_cost": unit_cost,
        "supplier_name": supplier_name,
        "supplier_lead_time_days": supplier_lead_time_days,
        "reorder_threshold_kg": reorder_threshold_kg,
    })
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection(db_path) as conn:
        with conn:
            conn.execute(
                "INSERT INTO feed_inventory (feed_name, available_stock_kg, daily_usage_kg, unit_cost, "
                "supplier_name, supplier_lead_time_days, reorder_threshold_kg, last_updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(feed_name) DO UPDATE SET "
                "available_stock_kg=excluded.available_stock_kg, "
                "daily_usage_kg=excluded.daily_usage_kg, unit_cost=excluded.unit_cost, "
                "supplier_name=excluded.supplier_name, "
                "supplier_lead_time_days=excluded.supplier_lead_time_days, "
                "reorder_threshold_kg=excluded.reorder_threshold_kg, "
                "last_updated=excluded.last_updated",
                (
                    item["feed_name"],
                    item["available_stock_kg"],
                    item["daily_usage_kg"],
                    item["unit_cost"],
                    item["supplier_name"],
                    item["supplier_lead_time_days"],
                    item["reorder_threshold_kg"],
                    timestamp,
                ),
            )
            row = conn.execute(
                "SELECT id FROM feed_inventory WHERE feed_name = ?", (item["feed_name"],)
            ).fetchone()
            return int(row["id"])


def get_feed_inventory(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return all inventory items in case-insensitive feed-name order."""
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM feed_inventory ORDER BY lower(feed_name), id"
        ).fetchall()
        return [dict(row) for row in rows]
