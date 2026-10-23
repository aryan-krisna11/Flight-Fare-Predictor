"""Train, evaluate and persist the fare model from collected API data."""
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flightfare.config import CURRENCY, DB_PATH, MIN_TRAINING_ROWS, MODEL_PATH, RESULTS_DIR
from flightfare.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, training_frame
from flightfare.store import load_fares


def _preprocessor(scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=3, sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
        ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
    ])


def candidate_models():
    # Fares are right-skewed, so every model learns log(price) and predicts back in rupees.
    def wrap(regressor, scale_numeric=False):
        pipeline = Pipeline([("preprocessor", _preprocessor(scale_numeric)), ("model", regressor)])
        return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)

    return {
        "Ridge Regression": wrap(Ridge(alpha=1.0), scale_numeric=True),
        "Random Forest": wrap(RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, max_features=0.5, random_state=42, n_jobs=-1)),
        "Gradient Boosting": wrap(HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, early_stopping=True, random_state=42)),
    }


def split(X, y, seen, test_size=0.2):
    """Hold out the most recent observations when data spans several days (honest for
    forecasting); fall back to a random split on the first day of collection."""
    if seen.dt.normalize().nunique() >= 2:
        order = np.argsort(seen.values, kind="stable")
        cut = int(len(order) * (1 - test_size))
        train_idx, test_idx = order[:cut], order[cut:]
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], "time-based (latest 20% held out)"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test, "random 80/20 (single collection day)"


def _lookups(X):
    """Typical durations and departure hours, used to fill gaps in a prediction request."""
    def median_map(keys, column):
        grouped = X.dropna(subset=[column]).groupby(keys)[column].median()
        return {(k if isinstance(k, tuple) else (k,)): float(v) for k, v in grouped.items()}

    return {
        "duration_by_route_airline_transfers": median_map(["route", "airline", "transfers"], "duration_min"),
        "duration_by_route_transfers": median_map(["route", "transfers"], "duration_min"),
        "duration_by_route": median_map(["route"], "duration_min"),
        "hour_by_route_airline": median_map(["route", "airline"], "dep_hour"),
        "hour_by_route": median_map(["route"], "dep_hour"),
        "airlines_by_route": {r: sorted(g["airline"].unique().tolist()) for r, g in X.groupby("route")},
    }


def train(db_path=DB_PATH, model_path=MODEL_PATH, results_dir=RESULTS_DIR, currency=CURRENCY, log=print):
    fares = load_fares(db_path, currency=currency)
    X, y, seen = training_frame(fares)
    log(f"Fare observations in store: {len(fares):,}; usable for training: {len(X):,}")
    if len(X) < MIN_TRAINING_ROWS:
        raise RuntimeError(
            f"Only {len(X)} usable fares; need at least {MIN_TRAINING_ROWS}. Run scripts/collect.py first."
        )

    X_train, X_test, y_train, y_test, split_kind = split(X, y, seen)
    log(f"Split: {split_kind}; train={len(X_train):,}, test={len(X_test):,}")

    results, fitted = [], {}
    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        result = {
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "MAPE": mean_absolute_percentage_error(y_test, predictions),
            "R2": r2_score(y_test, predictions),
        }
        log(f"  {name}: MAE=₹{result['MAE']:,.0f}  RMSE=₹{result['RMSE']:,.0f}  "
            f"MAPE={result['MAPE']:.1%}  R²={result['R2']:.3f}")
        results.append(result)
        fitted[name] = model

    results_df = pd.DataFrame(results)
    best = max(results, key=lambda r: r["R2"])
    best_model = fitted[best["Model"]]

    importance = permutation_importance(
        best_model, X_test, y_test, n_repeats=5, random_state=42,
        scoring="neg_mean_absolute_error", n_jobs=1,
    )
    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "MAE_increase_when_shuffled": importance.importances_mean,
    }).sort_values("MAE_increase_when_shuffled", ascending=False)

    # Refit on everything so the deployed model includes the newest prices.
    final_model = candidate_models()[best["Model"]]
    final_model.fit(X, y)

    bundle = {
        "model": final_model,
        "model_name": best["Model"],
        "feature_columns": FEATURE_COLUMNS,
        "metrics": best,
        "split": split_kind,
        "currency": currency,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(X)),
        "observed_from": seen.min().isoformat(),
        "observed_to": seen.max().isoformat(),
        "lookups": _lookups(X),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    results_df.to_csv(results_dir / "model_comparison.csv", index=False)
    importance_df.to_csv(results_dir / "feature_importance.csv", index=False)

    log(f"Best model: {best['Model']} (R²={best['R2']:.3f}, MAE=₹{best['MAE']:,.0f}); saved to {model_path}")
    return bundle
