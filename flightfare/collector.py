"""Pull current fares from the API into the SQLite store."""
import time
from datetime import date, datetime, timezone

from flightfare.api import TravelpayoutsError, parse_offer
from flightfare.store import insert_fares


def upcoming_months(count, today=None):
    today = today or date.today()
    months = []
    year, month = today.year, today.month
    for _ in range(count):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return months


def fetch_offers(client, origin, destination, departure_at):
    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw = client.prices_for_dates(origin, destination, departure_at)
    rows = [parse_offer(item, client.currency, collected_at) for item in raw]
    return [r for r in rows if r is not None], len(raw)


def collect(client, routes, months, db_path, pause_seconds=1.0, log=print):
    """Fetch every route x month; returns (rows_received, new_rows_stored, failed_requests)."""
    received = stored = failed = 0
    for origin, destination in routes:
        for month in months:
            try:
                rows, raw_count = fetch_offers(client, origin, destination, month)
            except (TravelpayoutsError, OSError, ValueError) as exc:
                failed += 1
                log(f"  {origin}-{destination} {month}: FAILED ({exc})")
                time.sleep(pause_seconds)
                continue
            new = insert_fares(rows, db_path)
            received += len(rows)
            stored += new
            skipped = raw_count - len(rows)
            log(f"  {origin}-{destination} {month}: {len(rows)} fares, {new} new"
                + (f", {skipped} unparseable" if skipped else ""))
            time.sleep(pause_seconds)
    return received, stored, failed
