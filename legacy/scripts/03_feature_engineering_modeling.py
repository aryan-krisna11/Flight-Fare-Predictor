import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "flight_pricing_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "flight_price_model.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VISUAL_DIR = os.path.join(BASE_DIR, "visuals")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Loading cleaned flight data...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# ---------------- Feature Engineering ----------------
if {"Source", "Destination"}.issubset(df.columns):
    df["Route"] = (
        df["Source"].astype(str).str.strip()
        + "-" + df["Destination"].astype(str).str.strip()
    )

if "Duration_Minutes" in df.columns:
    df["Duration_Hours"] = pd.to_numeric(df["Duration_Minutes"], errors="coerce") / 60.0

if "Departure_Date" in df.columns:
    dates = pd.to_datetime(df["Departure_Date"], errors="coerce")
    df["Month"] = dates.dt.month
    df["Is_Weekend"] = (dates.dt.dayofweek >= 5).astype(float)

if "Days_Before_Departure" in df.columns:
    days = pd.to_numeric(df["Days_Before_Departure"], errors="coerce")
    df["Booking_Urgency"] = pd.cut(
        days,
        bins=[-np.inf, 7, 15, 30, 60, np.inf],
        labels=["Last-minute", "Short", "Medium", "Long", "Very Long"]
    ).astype("object")


# ---------------- Prepare X and y ----------------
target = "Price"
drop_cols = [c for c in ["Flight_ID", "Departure_Date"] if c in df.columns]
X = df.drop(columns=[target] + drop_cols)
y = pd.to_numeric(df[target], errors="coerce")

valid = y.notna()
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

categorical_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
numeric_features = [c for c in X.columns if c not in categorical_features]

print(f"Categorical features: {len(categorical_features)}")
print(f"Numeric features: {len(numeric_features)}")

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, categorical_features),
    ("numeric", numeric_pipeline, numeric_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=20, max_depth=12, max_samples=0.5, max_features=0.5, random_state=42, n_jobs=1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=25, learning_rate=0.05, max_depth=3, random_state=42
    )
}

best_result = None
best_model = None
results = []
predictions_for_best = None

print("\nModel comparison:")
for name, model in models.items():
    print(f"Training {name}...")
    # Gradient Boosting requires a dense matrix; tree ensembles and linear regression can use the sparse matrix.
    if name == "Gradient Boosting":
        model.fit(X_train_processed.toarray(), y_train)
        predictions = model.predict(X_test_processed.toarray())
    else:
        model.fit(X_train_processed, y_train)
        predictions = model.predict(X_test_processed)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"{name}: MAE=₹{mae:,.0f}, RMSE=₹{rmse:,.0f}, R²={r2:.3f}")
    result = {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}
    results.append(result)

    if best_result is None or r2 > best_result["R2"]:
        best_result = result
        best_model = model
        predictions_for_best = predictions

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

# ---------------- Feature importance ----------------
# Aggregate one-hot encoded columns back to their original feature names.
feature_names = list(preprocessor.get_feature_names_out())
importance_df = pd.DataFrame({
    "Encoded_Feature": feature_names,
    "Importance": getattr(best_model, "feature_importances_", np.zeros(len(feature_names)))
})

importance_df["Original_Feature"] = importance_df["Encoded_Feature"].str.replace(
    r"^(categorical|numeric)__", "", regex=True
)
importance_df["Original_Feature"] = importance_df["Original_Feature"].str.split("_").str[0]

# Safer mapping: use known source feature names from transformed prefixes.
def original_name(encoded):
    clean = encoded.split("__", 1)[-1]
    if clean in {"Duration_Minutes", "Duration_Hours"} or clean.startswith("Duration_Minutes_") or clean.startswith("Duration_Hours_"):
        return "Duration"
    for col in categorical_features + numeric_features:
        if clean == col or clean.startswith(col + "_"):
            return col
    return clean

importance_df["Original_Feature"] = importance_df["Encoded_Feature"].map(original_name)
importance_grouped = (
    importance_df.groupby("Original_Feature", as_index=False)["Importance"]
    .sum().sort_values("Importance", ascending=False)
)
importance_grouped.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)

# ---------------- Save model bundle ----------------
bundle = {
    "model": best_model,
    "preprocessor": preprocessor,
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "feature_columns": X.columns.tolist(),
    "metrics": {
        "model": best_result["Model"],
        "MAE": best_result["MAE"],
        "RMSE": best_result["RMSE"],
        "R2": best_result["R2"]
    },
    "feature_importance": importance_grouped.to_dict("records")
}
joblib.dump(bundle, MODEL_PATH)

# ---------------- Evaluation visuals ----------------
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))
plt.bar(results_df["Model"], results_df["R2"])
plt.ylim(0, 1)
plt.title("Model Comparison by R²")
plt.ylabel("R² score")
plt.xlabel("")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "06_model_comparison_r2.png"), dpi=160)
plt.close()

plt.figure(figsize=(9, 6))
top_imp = importance_grouped.head(10).sort_values("Importance")
plt.barh(top_imp["Original_Feature"], top_imp["Importance"])
plt.gca().invert_yaxis()
plt.title("Top 10 Features Driving Random Forest Predictions")
plt.xlabel("Aggregated feature importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "07_feature_importance.png"), dpi=160)
plt.close()

pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions_for_best})
plt.figure(figsize=(8, 7))
sample_pred = pred_df.sample(min(8000, len(pred_df)), random_state=42)
plt.scatter(sample_pred["Actual"], sample_pred["Predicted"], alpha=0.35, s=12)
lims = [min(pred_df.min()), max(pred_df.max())]
plt.plot(lims, lims, linestyle="--")
plt.title("Actual vs Predicted Flight Prices")
plt.xlabel("Actual Price (₹)")
plt.ylabel("Predicted Price (₹)")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "08_actual_vs_predicted.png"), dpi=160)
plt.close()

print("\n----------------------------------------")
print(f"BEST MODEL: {best_result['Model']}")
print(f"MAE:  ₹{best_result['MAE']:,.2f}")
print(f"RMSE: ₹{best_result['RMSE']:,.2f}")
print(f"R²:   {best_result['R2']:.4f}")
print("----------------------------------------")
print(f"Model saved to: {MODEL_PATH}")
print(f"Results saved to: {RESULTS_DIR}")
