"""Collect live fares from the Travelpayouts Data API into data/fares.db.

Examples:
    python scripts/collect.py                          # all default routes, next 3 months
    python scripts/collect.py --routes DEL-BOM,BOM-DEL --months 2
    python scripts/collect.py --every-hours 6 --retrain  # keep collecting and retraining
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Windows consoles/pipes default to cp1252, which can't print the rupee sign.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from flightfare.api import TravelpayoutsClient  # noqa: E402
from flightfare.collector import collect, upcoming_months  # noqa: E402
from flightfare.config import DB_PATH, DEFAULT_ROUTES, get_token, parse_routes  # noqa: E402
from flightfare.model import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES),
                        help="Comma-separated IATA pairs, e.g. DEL-BOM,BOM-BLR (default: top Indian routes)")
    parser.add_argument("--months", type=int, default=3, help="How many departure months ahead to fetch")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds between API requests (rate limit)")
    parser.add_argument("--every-hours", type=float, default=0,
                        help="Repeat collection on this interval; 0 = run once")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model after each collection run")
    args = parser.parse_args()

    client = TravelpayoutsClient(get_token())
    routes = parse_routes(args.routes)

    while True:
        months = upcoming_months(args.months)
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] Collecting {len(routes)} routes x {len(months)} months "
              f"({', '.join(months)})")
        received, stored, failed = collect(client, routes, months, DB_PATH, pause_seconds=args.pause)
        print(f"Done: {received:,} fares received, {stored:,} new stored, {failed} failed requests -> {DB_PATH}")

        if args.retrain:
            try:
                train()
            except RuntimeError as exc:
                print(f"Skipping retrain: {exc}")

        if args.every_hours <= 0:
            break
        print(f"Next run in {args.every_hours} h (Ctrl+C to stop)")
        time.sleep(args.every_hours * 3600)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")
