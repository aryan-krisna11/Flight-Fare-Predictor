"""SQLite store of every fare observation collected from the API."""
import sqlite3
from contextlib import closing

import pandas as pd

from flightfare.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS fares (
    id INTEGER PRIMARY KEY,
    collected_at TEXT NOT NULL,
    collected_date TEXT NOT NULL,
    found_at TEXT,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    origin_airport TEXT NOT NULL DEFAULT '',
    destination_airport TEXT NOT NULL DEFAULT '',
    airline TEXT NOT NULL DEFAULT '',
    flight_number TEXT NOT NULL DEFAULT '',
    departure_at TEXT NOT NULL,
    transfers INTEGER,
    duration_min REAL,
    price REAL NOT NULL,
    currency TEXT NOT NULL,
    -- The same flight at the same price seen again on the same day is one observation.
    UNIQUE (origin, destination, departure_at, airline, flight_number, transfers, price, collected_date)
);
CREATE INDEX IF NOT EXISTS idx_fares_route ON fares (origin, destination, departure_at);
"""

COLUMNS = [
    "collected_at", "collected_date", "found_at", "origin", "destination", "origin_airport",
    "destination_airport", "airline", "flight_number", "departure_at", "transfers",
    "duration_min", "price", "currency",
]


def connect(db_path=DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def insert_fares(rows, db_path=DB_PATH):
    """Insert normalized rows; returns how many were new."""
    rows = [dict(r, collected_date=r["collected_at"][:10]) for r in rows]
    if not rows:
        return 0
    placeholders = ", ".join(f":{c}" for c in COLUMNS)
    sql = f"INSERT OR IGNORE INTO fares ({', '.join(COLUMNS)}) VALUES ({placeholders})"
    with closing(connect(db_path)) as conn:
        before = conn.total_changes
        conn.executemany(sql, rows)
        conn.commit()
        return conn.total_changes - before


def load_fares(db_path=DB_PATH, currency=None):
    if not db_path.exists():
        return pd.DataFrame(columns=COLUMNS)
    with closing(connect(db_path)) as conn:
        if currency:
            return pd.read_sql_query("SELECT * FROM fares WHERE currency = ?", conn, params=(currency,))
        return pd.read_sql_query("SELECT * FROM fares", conn)
