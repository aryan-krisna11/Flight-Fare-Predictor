import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "flight_pricing_cleaned.csv")
VISUAL_DIR = os.path.join(BASE_DIR, "visuals")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(VISUAL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nAverage price by travel class:")
print(df.groupby("Travel_Class")["Price"].mean().sort_values(ascending=False))

# 01 Price distribution
plt.figure(figsize=(10, 6))
plt.hist(df["Price"], bins=50, edgecolor="black", alpha=0.75)
plt.title("Flight Price Distribution")
plt.xlabel("Price (₹)")
plt.ylabel("Number of Flights")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "01_price_distribution.png"), dpi=160)
plt.close()

# 02 Price by travel class
plt.figure(figsize=(10, 6))
classes = [c for c in df["Travel_Class"].dropna().unique()]
plt.boxplot([df.loc[df["Travel_Class"] == c, "Price"] for c in classes], tick_labels=classes)
plt.title("Price by Travel Class")
plt.xlabel("Travel Class")
plt.ylabel("Price (₹)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "02_price_by_class.png"), dpi=160)
plt.close()

# 03 Distance vs price
plt.figure(figsize=(10, 6))
sample = df.sample(min(10000, len(df)), random_state=42)
plt.scatter(sample["Distance_km"], sample["Price"], alpha=0.35, s=12)
plt.title("Distance vs Flight Price")
plt.xlabel("Distance (km)")
plt.ylabel("Price (₹)")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "03_distance_vs_price.png"), dpi=160)
plt.close()

# 04 Average price by airline
airline_price = df.groupby("Airline", as_index=False)["Price"].mean().sort_values("Price", ascending=False)
plt.figure(figsize=(11, 6))
plt.barh(airline_price["Airline"], airline_price["Price"])
plt.gca().invert_yaxis()
plt.title("Average Flight Price by Airline")
plt.xlabel("Average Price (₹)")
plt.ylabel("Airline")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "04_price_by_airline.png"), dpi=160)
plt.close()

# 05 Average price by booking window
booking_bins = [-1, 7, 15, 30, 60, float("inf")]
booking_labels = ["0–7 days", "8–15 days", "16–30 days", "31–60 days", "61+ days"]
df["Booking_Window"] = pd.cut(
    pd.to_numeric(df["Days_Before_Departure"], errors="coerce"),
    bins=booking_bins, labels=booking_labels
)
booking_price = df.groupby("Booking_Window", observed=False)["Price"].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.plot(booking_price["Booking_Window"].astype(str), booking_price["Price"], marker="o")
plt.title("Average Flight Price vs Days Before Departure")
plt.xlabel("Days Before Departure")
plt.ylabel("Average Price (₹)")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "05_price_vs_booking_days.png"), dpi=160)
plt.close()

# 06 Average price by stops
stops_price = df.groupby("Total_Stops", as_index=False)["Price"].mean().sort_values("Total_Stops")
plt.figure(figsize=(9, 6))
plt.bar(stops_price["Total_Stops"].astype(str), stops_price["Price"])
plt.title("Average Flight Price by Number of Stops")
plt.xlabel("Number of Stops")
plt.ylabel("Average Price (₹)")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "09_price_by_stops.png"), dpi=160)
plt.close()

# Save compact EDA tables used in the README.
airline_price.to_csv(os.path.join(RESULTS_DIR, "airline_price_summary.csv"), index=False)
booking_price.to_csv(os.path.join(RESULTS_DIR, "booking_price_summary.csv"), index=False)
stops_price.to_csv(os.path.join(RESULTS_DIR, "stops_price_summary.csv"), index=False)

print("\nEDA charts saved in:", VISUAL_DIR)
