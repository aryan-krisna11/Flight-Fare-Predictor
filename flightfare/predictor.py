"""Load the trained model and predict fares for a route/date. UI-agnostic: a CLI or web app
can call FarePredictor directly."""
from datetime import date, datetime, timezone

import joblib
import pandas as pd

from flightfare.config import MODEL_PATH
from flightfare.features import build_features


class FarePredictor:
    def __init__(self, bundle):
        self.bundle = bundle
        self.model = bundle["model"]
        self.lookups = bundle["lookups"]

    @classmethod
    def load(cls, model_path=MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model at {model_path}. Run scripts/train.py first.")
        return cls(joblib.load(model_path))

    def airlines_for(self, origin, destination):
        return self.lookups["airlines_by_route"].get(f"{origin}-{destination}", [])

    def _default_duration(self, route, airline, transfers):
        lk = self.lookups
        for table, key in (
            ("duration_by_route_airline_transfers", (route, airline, float(transfers))),
            ("duration_by_route_transfers", (route, float(transfers))),
            ("duration_by_route", (route,)),
        ):
            if key in lk[table]:
                return lk[table][key]
        return None  # the model's median imputer fills it

    def _default_hour(self, route, airline):
        hour = self.lookups["hour_by_route_airline"].get((route, airline))
        if hour is None:
            hour = self.lookups["hour_by_route"].get((route,), 12)
        return int(round(hour))

    def predict(self, requests, observed_at=None):
        """Predict prices for a list of dicts with keys origin, destination, departure_date
        (date/'YYYY-MM-DD'), and optional airline, transfers, departure_hour, duration_min.
        Returns a DataFrame of the filled-in requests with a predicted_price column."""
        observed_at = observed_at or datetime.now(timezone.utc)
        rows = []
        for req in requests:
            origin, destination = req["origin"].upper(), req["destination"].upper()
            route = f"{origin}-{destination}"
            airline = (req.get("airline") or "UNKNOWN").upper()
            transfers = int(req.get("transfers", 0) or 0)
            hour = req.get("departure_hour")
            hour = self._default_hour(route, airline) if hour is None else int(hour)
            duration = req.get("duration_min")
            if duration is None:
                duration = self._default_duration(route, airline, transfers)
            departure_date = req["departure_date"]
            if isinstance(departure_date, str):
                departure_date = date.fromisoformat(departure_date)
            rows.append({
                "origin": origin,
                "destination": destination,
                "airline": airline,
                "transfers": transfers,
                "duration_min": duration,
                "departure_at": f"{departure_date.isoformat()}T{hour:02d}:00:00+05:30",
                "found_at": None,
                "collected_at": observed_at.isoformat(),
            })
        frame = pd.DataFrame(rows)
        frame["predicted_price"] = self.model.predict(build_features(frame))
        return frame

    def predict_route(self, origin, destination, departure_date, transfers=0, airline=None,
                      departure_hour=None, observed_at=None):
        """Predict for one airline, or for every airline seen on the route when airline is None."""
        airlines = [airline] if airline else (self.airlines_for(origin, destination) or [None])
        return self.predict([
            {"origin": origin, "destination": destination, "departure_date": departure_date,
             "airline": a, "transfers": transfers, "departure_hour": departure_hour}
            for a in airlines
        ], observed_at=observed_at).sort_values("predicted_price").reset_index(drop=True)
