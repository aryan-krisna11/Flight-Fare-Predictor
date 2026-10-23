"""Client for the free Travelpayouts (Aviasales) Data API.

Prices are the cheapest fares Aviasales users found in roughly the last 48 hours, so they
are near-live market prices, not a bookable quote at this exact second.
"""
import time
from datetime import datetime, timezone

import requests

from flightfare.config import CURRENCY, MARKET

PRICES_FOR_DATES_URL = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"


class TravelpayoutsError(RuntimeError):
    pass


class TravelpayoutsClient:
    def __init__(self, token, currency=CURRENCY, market=MARKET, timeout=20,
                 max_retries=3, backoff_seconds=5.0, session=None):
        self.token = token
        self.currency = currency
        self.market = market
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = session or requests.Session()

    def prices_for_dates(self, origin, destination, departure_at, one_way=True,
                         direct=False, limit=1000, page=1):
        """Return raw offer dicts. `departure_at` is 'YYYY-MM' (whole month) or 'YYYY-MM-DD'."""
        params = {
            "origin": origin,
            "destination": destination,
            "departure_at": departure_at,
            "one_way": str(one_way).lower(),
            "direct": str(direct).lower(),
            "currency": self.currency,
            "market": self.market,
            "sorting": "price",
            "unique": "false",
            "limit": limit,
            "page": page,
        }
        headers = {"X-Access-Token": self.token, "Accept-Encoding": "gzip, deflate"}

        for attempt in range(self.max_retries + 1):
            response = self.session.get(PRICES_FOR_DATES_URL, params=params,
                                        headers=headers, timeout=self.timeout)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.max_retries:
                    raise TravelpayoutsError(
                        f"HTTP {response.status_code} after {self.max_retries} retries: {response.text[:200]}"
                    )
                time.sleep(self.backoff_seconds * (2 ** attempt))
                continue
            if response.status_code in (401, 403):
                raise TravelpayoutsError(f"HTTP {response.status_code}: token rejected. Check TRAVELPAYOUTS_TOKEN.")
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success", False):
                raise TravelpayoutsError(f"API error: {payload.get('error') or payload}")
            return payload.get("data") or []
        return []


def _first(item, *keys):
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def parse_offer(item, currency=CURRENCY, collected_at=None):
    """Normalize one API offer into a fares-table row, or None if it is unusable.

    Accepts both v3 field names (price, transfers, duration_to) and the older v2 names
    (value, number_of_changes, depart_date) so a schema drift doesn't silently drop data.
    """
    price = _first(item, "price", "value")
    departure_at = _first(item, "departure_at", "depart_date")
    origin = _first(item, "origin")
    destination = _first(item, "destination")
    if price is None or departure_at is None or not origin or not destination:
        return None
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None

    transfers = _first(item, "transfers", "number_of_changes")
    duration = _first(item, "duration_to", "duration")
    collected_at = collected_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    return {
        "collected_at": collected_at,
        "found_at": _first(item, "found_at"),
        "origin": str(origin).upper(),
        "destination": str(destination).upper(),
        "origin_airport": str(_first(item, "origin_airport") or ""),
        "destination_airport": str(_first(item, "destination_airport") or ""),
        "airline": str(_first(item, "airline") or ""),
        "flight_number": str(_first(item, "flight_number") or ""),
        "departure_at": str(departure_at),
        "transfers": int(transfers) if transfers is not None else None,
        "duration_min": float(duration) if duration is not None else None,
        "price": price,
        "currency": currency.lower(),
    }
