import os
from itertools import permutations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "fares.db"
MODEL_PATH = BASE_DIR / "models" / "fare_model.pkl"
RESULTS_DIR = BASE_DIR / "results"
ENV_PATH = BASE_DIR / ".env"

CURRENCY = "inr"
MARKET = "in"

# IATA city codes for India's busiest domestic markets. Every ordered pair is collected by default.
DEFAULT_CITIES = ["DEL", "BOM", "BLR", "HYD", "MAA", "CCU", "PNQ", "AMD", "GOI"]
DEFAULT_ROUTES = [f"{a}-{b}" for a, b in permutations(DEFAULT_CITIES, 2)]

# Training is refused below this many observed fares; the model would be noise.
MIN_TRAINING_ROWS = 200


def load_env(path=ENV_PATH):
    """Load KEY=VALUE lines from .env into os.environ without overriding real env vars."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_token():
    load_env()
    token = os.environ.get("TRAVELPAYOUTS_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "TRAVELPAYOUTS_TOKEN is not set. Get a free token at https://www.travelpayouts.com "
            "(Profile -> API token) and put it in .env as TRAVELPAYOUTS_TOKEN=..."
        )
    return token


def parse_routes(text):
    routes = []
    for part in text.split(","):
        part = part.strip().upper()
        if not part:
            continue
        origin, sep, destination = part.partition("-")
        if not sep or len(origin) != 3 or len(destination) != 3:
            raise ValueError(f"Invalid route '{part}'. Use IATA pairs like DEL-BOM.")
        routes.append((origin, destination))
    return routes
