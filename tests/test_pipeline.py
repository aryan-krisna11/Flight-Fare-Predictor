"""End-to-end tests with a fake HTTP session; no network or token required.

Run: python -m unittest discover -s tests -v
"""
import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flightfare import model as model_module  # noqa: E402
from flightfare.api import TravelpayoutsClient, TravelpayoutsError, parse_offer  # noqa: E402
from flightfare.collector import collect, upcoming_months  # noqa: E402
from flightfare.config import parse_routes  # noqa: E402
from flightfare.features import build_features, training_frame  # noqa: E402
from flightfare.predictor import FarePredictor  # noqa: E402
from flightfare.store import insert_fares, load_fares  # noqa: E402

AIRLINES = {"6E": 1.0, "AI": 1.25, "UK": 1.4, "SG": 0.95}
BASE_FARE = {"DEL-BOM": 4500, "BOM-DEL": 4600, "DEL-BLR": 5500, "BLR-DEL": 5400}


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")


class FakeSession:
    """Generates v3-shaped prices_for_dates responses with a learnable price signal."""

    def __init__(self, responses=None, seed=0):
        self.responses = list(responses or [])
        self.calls = []
        self.rng = np.random.default_rng(seed)

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(params)
        if self.responses:
            return self.responses.pop(0)
        return FakeResponse(200, {"success": True, "data": self._offers(params), "currency": "inr"})

    def _offers(self, params):
        route = f"{params['origin']}-{params['destination']}"
        year, month = map(int, params["departure_at"][:7].split("-"))
        today = date.today()
        offers = []
        for _ in range(60):
            day = int(self.rng.integers(1, 29))
            dep = date(year, month, day)
            if dep <= today:
                continue
            airline = self.rng.choice(list(AIRLINES))
            hour = int(self.rng.integers(5, 23))
            transfers = int(self.rng.choice([0, 0, 0, 1]))
            days_out = (dep - today).days
            price = BASE_FARE[route] * AIRLINES[airline] * (1 + 0.6 * transfers)
            price *= 1 + 1.5 * np.exp(-days_out / 10)          # last-minute premium
            price *= 1.15 if dep.weekday() >= 5 else 1.0
            price *= self.rng.normal(1, 0.05)
            offers.append({
                "origin": params["origin"], "destination": params["destination"],
                "origin_airport": params["origin"], "destination_airport": params["destination"],
                "price": round(price), "airline": airline, "flight_number": str(self.rng.integers(100, 9999)),
                "departure_at": f"{dep.isoformat()}T{hour:02d}:15:00+05:30", "return_at": "",
                "transfers": transfers, "return_transfers": 0,
                "duration": 130 + 150 * transfers, "duration_to": 130 + 150 * transfers,
                "duration_back": 0, "link": "/search/x",
            })
        return offers


class ParseOfferTests(unittest.TestCase):
    def test_v3_fields(self):
        row = parse_offer({"origin": "del", "destination": "BOM", "price": 4321, "airline": "6E",
                           "flight_number": 2133, "departure_at": "2026-11-15T06:00:00+05:30",
                           "transfers": 0, "duration_to": 135}, collected_at="2026-10-23T10:00:00+00:00")
        self.assertEqual(row["origin"], "DEL")
        self.assertEqual(row["price"], 4321.0)
        self.assertEqual(row["flight_number"], "2133")
        self.assertEqual(row["duration_min"], 135.0)

    def test_v2_fields_fallback(self):
        row = parse_offer({"origin": "DEL", "destination": "BOM", "value": 5000,
                           "depart_date": "2026-11-15", "number_of_changes": 1,
                           "found_at": "2026-10-22T08:00:00"})
        self.assertEqual(row["price"], 5000.0)
        self.assertEqual(row["transfers"], 1)
        self.assertEqual(row["departure_at"], "2026-11-15")

    def test_rejects_unusable(self):
        self.assertIsNone(parse_offer({"origin": "DEL", "destination": "BOM", "price": 0, "departure_at": "2026-11-15"}))
        self.assertIsNone(parse_offer({"origin": "DEL", "destination": "BOM", "departure_at": "2026-11-15"}))


class ClientTests(unittest.TestCase):
    def test_api_error_raises(self):
        session = FakeSession([FakeResponse(200, {"success": False, "error": "bad origin"})])
        client = TravelpayoutsClient("t", session=session)
        with self.assertRaises(TravelpayoutsError):
            client.prices_for_dates("XXX", "BOM", "2026-11")

    def test_retries_on_429(self):
        session = FakeSession([FakeResponse(429, {}), FakeResponse(200, {"success": True, "data": [{"x": 1}]})])
        client = TravelpayoutsClient("t", session=session, backoff_seconds=0)
        self.assertEqual(client.prices_for_dates("DEL", "BOM", "2026-11"), [{"x": 1}])
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[0]["currency"], "inr")

    def test_bad_token(self):
        client = TravelpayoutsClient("t", session=FakeSession([FakeResponse(401, {})]))
        with self.assertRaises(TravelpayoutsError):
            client.prices_for_dates("DEL", "BOM", "2026-11")


class HelperTests(unittest.TestCase):
    def test_upcoming_months_wraps_year(self):
        self.assertEqual(upcoming_months(3, date(2026, 11, 5)), ["2026-11", "2026-12", "2027-01"])

    def test_parse_routes(self):
        self.assertEqual(parse_routes("del-bom, BOM-BLR"), [("DEL", "BOM"), ("BOM", "BLR")])
        with self.assertRaises(ValueError):
            parse_routes("DELHI-BOM")

    def test_days_before_departure_uses_timezone(self):
        import pandas as pd
        frame = pd.DataFrame([{"origin": "DEL", "destination": "BOM", "airline": "6E", "transfers": 0,
                               "duration_min": 130, "departure_at": "2026-11-15T05:30:00+05:30",
                               "found_at": None, "collected_at": "2026-11-14T00:00:00+00:00"}])
        features = build_features(frame)
        self.assertAlmostEqual(features.loc[0, "days_before_departure"], 1.0)
        self.assertEqual(features.loc[0, "dep_hour"], 5)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "fares.db"
        self.model_path = self.tmp / "model.pkl"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_collect_train_predict(self):
        client = TravelpayoutsClient("t", session=FakeSession(seed=1))
        routes = parse_routes(",".join(BASE_FARE))
        months = upcoming_months(3)
        received, stored, failed = collect(client, routes, months, self.db, pause_seconds=0, log=lambda *_: None)
        self.assertEqual(failed, 0)
        self.assertGreater(stored, 300)

        # Re-inserting identical observations on the same day must not duplicate.
        fares = load_fares(self.db)
        rows = fares.drop(columns=["id", "collected_date"]).to_dict("records")
        self.assertEqual(insert_fares(rows, self.db), 0)

        X, y, _ = training_frame(fares)
        self.assertEqual(len(X), len(fares))

        bundle = model_module.train(db_path=self.db, model_path=self.model_path,
                                    results_dir=self.tmp, log=lambda *_: None)
        self.assertGreater(bundle["metrics"]["R2"], 0.6)
        self.assertTrue((self.tmp / "model_comparison.csv").exists())

        predictor = FarePredictor.load(self.model_path)
        departure = date.today() + timedelta(days=30)
        result = predictor.predict_route("DEL", "BOM", departure)
        self.assertEqual(set(result["airline"]), set(AIRLINES))
        self.assertTrue((result["predicted_price"] > 0).all())

        # The model should learn the synthetic signal: AI/UK pricier than 6E, last-minute pricier.
        by_airline = dict(zip(result["airline"], result["predicted_price"]))
        self.assertGreater(by_airline["UK"], by_airline["6E"])
        soon = predictor.predict_route("DEL", "BOM", date.today() + timedelta(days=2), airline="6E")
        later = predictor.predict_route("DEL", "BOM", date.today() + timedelta(days=45), airline="6E")
        self.assertGreater(soon["predicted_price"][0], later["predicted_price"][0])

    def test_train_refuses_tiny_dataset(self):
        insert_fares([parse_offer({"origin": "DEL", "destination": "BOM", "price": 4000,
                                   "departure_at": "2027-01-10T06:00:00+05:30", "airline": "6E"},
                                  collected_at=datetime.now(timezone.utc).isoformat())], self.db)
        with self.assertRaises(RuntimeError):
            model_module.train(db_path=self.db, model_path=self.model_path, results_dir=self.tmp, log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
