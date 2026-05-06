"""
data_logger.py
Real-time solar wind data ingestion from NOAA API with fallback simulation.
Continuously appends to solar_data.csv at a configurable interval.
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NOAA_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"
NOAA_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"
DATA_PATH = os.path.join(os.path.dirname(__file__), "solar_data.csv")
FETCH_INTERVAL = 60  # seconds

COLUMNS = ["timestamp", "speed", "density", "bz", "bt", "temperature"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _init_csv():
    if not os.path.exists(DATA_PATH):
        pd.DataFrame(columns=COLUMNS).to_csv(DATA_PATH, index=False)
        logger.info("Created solar_data.csv")


def _simulate_row(prev: dict | None = None) -> dict:
    """Generate physically plausible solar wind values using a random walk."""
    if prev is None:
        speed = np.random.uniform(350, 500)
        density = np.random.uniform(3, 10)
        bz = np.random.uniform(-5, 5)
        bt = np.random.uniform(2, 10)
        temperature = np.random.uniform(50000, 150000)
    else:
        speed = float(prev["speed"]) + np.random.normal(0, 5)
        speed = np.clip(speed, 280, 900)
        density = float(prev["density"]) + np.random.normal(0, 0.5)
        density = np.clip(density, 0.5, 30)
        bz = float(prev["bz"]) + np.random.normal(0, 2)
        bz = np.clip(bz, -40, 20)
        bt = abs(bz) + np.random.uniform(0, 5)
        temperature = float(prev["temperature"]) + np.random.normal(0, 3000)
        temperature = np.clip(temperature, 10000, 500000)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "speed": round(speed, 2),
        "density": round(density, 3),
        "bz": round(bz, 3),
        "bt": round(bt, 3),
        "temperature": round(temperature, 0),
    }


def _fetch_noaa() -> dict | None:
    """Try to pull the latest reading from NOAA SWPC. Returns None on failure."""
    try:
        mag_resp = requests.get(NOAA_URL, timeout=10)
        plasma_resp = requests.get(NOAA_PLASMA_URL, timeout=10)

        if mag_resp.status_code != 200 or plasma_resp.status_code != 200:
            return None

        mag_data = mag_resp.json()
        plasma_data = plasma_resp.json()

        # NOAA returns list-of-lists; index 0 is the header row
        mag_latest = mag_data[-1]    # [time_tag, bx, by, bz, lon, lat, bt]
        plasma_latest = plasma_data[-1]  # [time_tag, density, speed, temperature]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "speed": float(plasma_latest[2]) if plasma_latest[2] != "-9999.99" else None,
            "density": float(plasma_latest[1]) if plasma_latest[1] != "-9999.99" else None,
            "bz": float(mag_latest[3]) if mag_latest[3] != "-9999.99" else None,
            "bt": float(mag_latest[6]) if mag_latest[6] != "-9999.99" else None,
            "temperature": float(plasma_latest[3]) if plasma_latest[3] != "-9999.99" else None,
        }
    except Exception as e:
        logger.warning(f"NOAA fetch failed: {e}")
        return None


def _append_row(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(DATA_PATH, mode="a", header=not os.path.exists(DATA_PATH), index=False)


def get_latest_rows(n: int = 100) -> pd.DataFrame:
    """Read the last n rows from the CSV for API/model use."""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(DATA_PATH)
    return df.tail(n).reset_index(drop=True)


# ── main loop ─────────────────────────────────────────────────────────────────

def run_logger(simulate: bool = False):
    """
    Main ingestion loop.
    Args:
        simulate: Force simulation even if NOAA is reachable.
    """
    _init_csv()
    prev_row = None

    # Seed historical data if CSV is fresh
    existing = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()
    if len(existing) < 200:
        logger.info("Seeding 200 historical rows for model training...")
        for _ in range(200):
            row = _simulate_row(prev_row)
            _append_row(row)
            prev_row = row
        logger.info("Seeding complete.")

    logger.info("Starting real-time ingestion loop...")
    while True:
        row = None
        if not simulate:
            row = _fetch_noaa()

        if row is None or any(v is None for v in row.values()):
            logger.info("Using simulated data.")
            row = _simulate_row(prev_row)

        _append_row(row)
        prev_row = row
        logger.info(f"Logged: speed={row['speed']} bz={row['bz']} density={row['density']}")
        time.sleep(FETCH_INTERVAL)


if __name__ == "__main__":
    run_logger(simulate=False)