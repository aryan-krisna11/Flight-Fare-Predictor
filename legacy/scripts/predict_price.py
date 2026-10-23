import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "flight_price_model.pkl")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
preprocessor = bundle["preprocessor"]
feature_columns = bundle["feature_columns"]

# ---------------- Example flight ----------------
new_flight = pd.DataFrame([{
    "Airline": "Indigo",
    "Source": "Mumbai",
    "Destination": "Delhi",
    "Total_Stops": 0,
    "Distance_km": 1150,
    "Travel_Class": "Economy",
    "Days_Before_Departure": 25,
    "Season": "Winter",
    "Weekday": "Monday",
    "Aircraft_Type": "Airbus A320",
    "Booking_Channel": "Website",
    "Passenger_Count": 1,
    "Duration_Minutes": 126,
    "Departure_Hour": 8,
    "Departure_Time_Of_Day": "Morning",
    "Route": "Mumbai-Delhi",
    "Duration_Hours": 2.1,
    "Month": 12,
    "Is_Weekend": 0,
    "Is_International": 0,
    "Booking_Urgency": "Medium"
}])

# Match training columns exactly.
new_flight = new_flight.reindex(columns=feature_columns)

X_new = preprocessor.transform(new_flight)
predicted_price = model.predict(X_new)[0]

print(f"Predicted flight price: ₹{predicted_price:,.2f}")
