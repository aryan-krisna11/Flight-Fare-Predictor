"""Feature engineering shared by training and live prediction, so both see identical inputs."""
import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["origin", "destination", "route", "airline"]
NUMERIC_FEATURES = [
    "transfers", "duration_min", "days_before_departure", "dep_hour",
    "dep_dayofweek", "dep_month", "dep_day", "is_weekend",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _local_departure(series):
    # departure_at carries the airport's local offset (e.g. +05:30); keep the local clock time.
    return pd.to_datetime(series.astype(str).str.slice(0, 19), errors="coerce")


def _utc(series):
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def observed_at(df):
    """When the price was seen: the API's found_at if present, else our collection time."""
    found = _utc(df["found_at"]) if "found_at" in df else pd.Series(pd.NaT, index=df.index)
    return found.fillna(_utc(df["collected_at"]))


def build_features(df):
    """Return model inputs for rows with origin, destination, airline, transfers,
    duration_min, departure_at and found_at/collected_at."""
    out = pd.DataFrame(index=df.index)
    out["origin"] = df["origin"].astype(str).str.upper()
    out["destination"] = df["destination"].astype(str).str.upper()
    out["route"] = out["origin"] + "-" + out["destination"]
    out["airline"] = df["airline"].fillna("").astype(str).replace("", "UNKNOWN")

    out["transfers"] = pd.to_numeric(df["transfers"], errors="coerce")
    out["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")

    departure_local = _local_departure(df["departure_at"])
    departure_utc = _utc(df["departure_at"])
    # Dates without a time/offset parse as naive; treat them as IST.
    departure_utc = departure_utc.fillna(
        departure_local.dt.tz_localize("Asia/Kolkata", nonexistent="NaT", ambiguous="NaT").dt.tz_convert("UTC")
    )
    seen = observed_at(df)
    out["days_before_departure"] = (departure_utc - seen).dt.total_seconds() / 86400.0

    out["dep_hour"] = departure_local.dt.hour
    out["dep_dayofweek"] = departure_local.dt.dayofweek
    out["dep_month"] = departure_local.dt.month
    out["dep_day"] = departure_local.dt.day
    out["is_weekend"] = (departure_local.dt.dayofweek >= 5).astype(float)
    out.loc[departure_local.isna(), "is_weekend"] = np.nan
    return out[FEATURE_COLUMNS]


def training_frame(fares):
    """Features + target for training, dropping unusable and stale observations."""
    if fares.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=float), pd.Series(dtype="datetime64[ns, UTC]")
    X = build_features(fares)
    y = pd.to_numeric(fares["price"], errors="coerce")
    seen = observed_at(fares)
    # A price seen after departure (or with an unparseable date) can't teach booking-window effects.
    valid = y.notna() & (y > 0) & X["days_before_departure"].notna() & (X["days_before_departure"] >= 0)
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), seen.loc[valid].reset_index(drop=True)
