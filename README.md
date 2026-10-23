# Flight Fare Predictor (live data)

Collects real flight fares from the free **Travelpayouts (Aviasales) Data API**, trains a regression
model on them, and predicts fares for a route and date while comparing the prediction with the
current market price.

> The previous project, which used a static synthetic CSV (87,692 rows), is archived unchanged in
> [`legacy/`](legacy/). Its scripts still run from there, e.g. `python legacy\scripts\predict_price.py`.

## How "real time" works here

```text
 Travelpayouts API ──collect.py──▶ data/fares.db (SQLite, grows every run)
                                         │
                                    train.py ──▶ models/fare_model.pkl
                                         │
 predict.py ── live fares for the date ──┴──▶ model estimate + live comparison
              (also saved to fares.db)
```

- **Data**: the API returns the cheapest fares that Aviasales users found in the **last ~48 hours**.
  These are recent market prices, not a bookable quote at this exact second.
- **Training data accumulates**: every collection run (and every live lookup in `predict.py`) adds
  observations. The model gets better as it sees more days of prices, especially for booking-window
  effects. Expect weak results with less than a few days of collection.
- **Retraining**: run `train.py`, or `collect.py --every-hours 6 --retrain` to collect and retrain
  continuously.

## Setup

1. Create a free Travelpayouts account at https://www.travelpayouts.com and copy your API token
   (Profile → API token).
2. Copy `.env.example` to `.env` and paste in the token. `.env` is git-ignored.

   ```text
   TRAVELPAYOUTS_TOKEN=your_token_here
   ```
3. Install the dependencies: `pip install -r requirements.txt`

## Usage

```powershell
# 1. Collect fares: 72 routes between 9 major Indian cities, next 3 months (~216 requests)
python scripts\collect.py
python scripts\collect.py --routes DEL-BOM,BOM-DEL --months 2      # a subset

# 2. Train on everything collected so far
python scripts\train.py

# 3. Predict a fare (fetches live fares for that date and compares them)
python scripts\predict.py --from DEL --to BOM --date 2026-11-15
python scripts\predict.py --from BLR --to DEL --date 2026-12-20 --airline 6E --hour 7
python scripts\predict.py --from DEL --to BOM --date 2026-11-15 --offline   # model only

# Keep collecting every 6 h and retraining (leave running, Ctrl+C to stop)
python scripts\collect.py --every-hours 6 --retrain
```

To collect on a schedule without keeping a terminal open, use Windows Task Scheduler:

```powershell
schtasks /Create /SC HOURLY /MO 6 /TN FlightFareCollect /TR "python D:\projects\flight-fare-predictor\scripts\collect.py --retrain"
```

Codes are IATA **city** codes (DEL, BOM, BLR, HYD, MAA, CCU, PNQ, AMD, GOI, ...). Airlines are IATA
airline codes (6E IndiGo, AI Air India, SG SpiceJet, QP Akasa, IX Air India Express, ...).

## Model

| Feature | Source |
|---|---|
| origin, destination, route | API |
| airline, transfers, duration_min | API |
| days_before_departure | departure time − when the price was observed |
| dep_hour, dep_dayofweek, dep_month, dep_day, is_weekend | departure time (local) |

- Three candidates: Ridge Regression, Random Forest and Histogram Gradient Boosting. All of them
  learn `log(price)`, because fares are right-skewed. The best one by test R² is kept.
- **Evaluation split**: once the data spans two or more days, the newest 20% of observations is held
  out (time-based), which measures how well the model forecasts prices it hasn't seen yet. On the
  first day it falls back to a random 80/20 split.
- After evaluation, the chosen model is refit on all data so that deployed predictions include the
  latest prices.
- Outputs: `results/model_comparison.csv` (MAE, RMSE, MAPE, R²) and `results/feature_importance.csv`
  (permutation importance: how much MAE rises when a feature is shuffled).

When a prediction request leaves out duration or departure hour, typical values for that
route and airline from the training data are used. Routes the model has never seen are flagged
as unreliable.

## Code layout

```text
flightfare/          reusable package (a future browser UI should import this)
  config.py          paths, default routes, .env token loading
  api.py             Travelpayouts client + response normalization
  store.py           SQLite fare store (deduplicated observations)
  collector.py       route x month collection loop
  features.py        feature engineering shared by training and prediction
  model.py           training, evaluation, model bundle
  predictor.py       FarePredictor.load().predict_route(...)
scripts/             CLI entry points: collect.py, train.py, predict.py
tests/               offline tests with a simulated API (no token needed)
legacy/              archived synthetic-data project
```

Run the tests with `python -m unittest discover -s tests -v`.

## Limitations

- Prices are cached search results (≤ ~48 h old) for economy one-way tickets, so they can differ
  from the price at checkout. There are no seat-availability or fare-class details.
- Coverage depends on what Aviasales users have searched. Popular routes have many fares; thin
  routes may have few or none.
- The free API has per-minute rate limits; `collect.py --pause` sets the delay between requests
  (default 1 s).
