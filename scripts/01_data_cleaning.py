import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "flight_pricing_dataset.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "flight_pricing_cleaned.csv")

print("Loading raw flight data...")
df = pd.read_csv(RAW_PATH)
raw_rows, raw_cols = df.shape
print(f"Raw dataset shape: {df.shape}")

# Standardize column names and string values.
df.columns = [c.strip() for c in df.columns]
object_cols = df.select_dtypes(include="object").columns
for col in object_cols:
    df[col] = df[col].astype("string").str.strip()

# Convert known numeric columns safely.
numeric_cols = [
    "Total_Stops", "Distance_km", "Days_Before_Departure",
    "Passenger_Count", "Price", "Duration_Minutes", "Departure_Hour"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Parse dates consistently.
if "Departure_Date" in df.columns:
    df["Departure_Date"] = pd.to_datetime(df["Departure_Date"], errors="coerce")

# Remove exact duplicate rows.
before_dupes = len(df)
df = df.drop_duplicates().copy()
removed_dupes = before_dupes - len(df)

# Remove rows with invalid target values.
if "Price" not in df.columns:
    raise ValueError("Price column is missing from the dataset.")
invalid_price = df["Price"].isna() | (df["Price"] <= 0)
removed_invalid_price = int(invalid_price.sum())
df = df.loc[~invalid_price].copy()

# Remove impossible negative values where applicable.
for col in ["Distance_km", "Days_Before_Departure", "Passenger_Count", "Duration_Minutes"]:
    if col in df.columns:
        invalid = df[col].notna() & (df[col] < 0)
        df.loc[invalid, col] = pd.NA

# Keep date formatting consistent for downstream feature engineering.
if "Departure_Date" in df.columns:
    df["Departure_Date"] = df["Departure_Date"].dt.strftime("%Y-%m-%d")

os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
df.to_csv(CLEAN_PATH, index=False)

missing_total = int(df.isna().sum().sum())
print("\nCleaning summary")
print("----------------")
print(f"Rows before cleaning: {raw_rows:,}")
print(f"Duplicate rows removed: {removed_dupes:,}")
print(f"Invalid/non-positive prices removed: {removed_invalid_price:,}")
print(f"Rows after cleaning: {len(df):,}")
print(f"Remaining missing cells: {missing_total:,}")
print(f"Cleaned data saved to: {CLEAN_PATH}")
