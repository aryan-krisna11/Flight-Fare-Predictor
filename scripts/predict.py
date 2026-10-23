"""Predict a fare in real time and compare it with the live market price.

Examples:
    python scripts/predict.py --from DEL --to BOM --date 2026-11-15
    python scripts/predict.py --from BLR --to DEL --date 2026-12-20 --airline 6E --hour 7
    python scripts/predict.py --from DEL --to BOM --date 2026-11-15 --offline   # model only
"""
import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Windows consoles/pipes default to cp1252, which can't print the rupee sign.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from flightfare.api import TravelpayoutsClient, TravelpayoutsError  # noqa: E402
from flightfare.collector import fetch_offers  # noqa: E402
from flightfare.config import DB_PATH, get_token  # noqa: E402
from flightfare.predictor import FarePredictor  # noqa: E402
from flightfare.store import insert_fares  # noqa: E402


def stops_label(transfers):
    if transfers is None:
        return "?"
    return "nonstop" if transfers == 0 else f"{transfers} stop" + ("s" if transfers > 1 else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="origin", required=True, help="Origin IATA code, e.g. DEL")
    parser.add_argument("--to", dest="destination", required=True, help="Destination IATA code, e.g. BOM")
    parser.add_argument("--date", required=True, type=date.fromisoformat, help="Departure date YYYY-MM-DD")
    parser.add_argument("--airline", help="IATA airline code, e.g. 6E, AI, UK. Default: every airline on the route")
    parser.add_argument("--stops", type=int, default=0, help="Number of stops (default 0)")
    parser.add_argument("--hour", type=int, help="Departure hour 0-23 (default: typical for the route)")
    parser.add_argument("--offline", action="store_true", help="Skip the live API lookup")
    args = parser.parse_args()

    origin, destination = args.origin.upper(), args.destination.upper()
    days_out = (args.date - date.today()).days
    if days_out < 0:
        sys.exit("Error: departure date is in the past.")
    print(f"{origin} -> {destination}, departing {args.date} ({days_out} days from today)\n")

    live = []
    if not args.offline:
        try:
            client = TravelpayoutsClient(get_token())
            live, _ = fetch_offers(client, origin, destination, args.date.isoformat())
            # Every live lookup also becomes training data for the next retrain.
            insert_fares(live, DB_PATH)
        except (RuntimeError, TravelpayoutsError, OSError) as exc:
            print(f"Live lookup unavailable: {exc}\n")

    if live:
        live.sort(key=lambda r: r["price"])
        print("Live market fares (Aviasales, found in the last ~48 h):")
        for r in live[:10]:
            dep_time = r["departure_at"][11:16]
            print(f"  {r['airline']:<3} {r['flight_number']:<6} {dep_time}  "
                  f"{stops_label(r['transfers']):<8}  ₹{r['price']:,.0f}")
        print()
    elif not args.offline:
        print("No live fares found for this exact date.\n")

    try:
        predictor = FarePredictor.load()
    except FileNotFoundError as exc:
        sys.exit(f"Error: {exc}")

    meta = predictor.bundle
    predictions = predictor.predict_route(
        origin, destination, args.date, transfers=args.stops,
        airline=args.airline, departure_hour=args.hour, observed_at=datetime.now(timezone.utc),
    )
    print(f"Model estimate ({meta['model_name']}, trained {meta['trained_at'][:16]} on "
          f"{meta['n_rows']:,} fares, test MAE ₹{meta['metrics']['MAE']:,.0f}):")
    if not predictor.airlines_for(origin, destination):
        print("  (route not in training data; estimate is extrapolated and unreliable)")
    for row in predictions.itertuples():
        print(f"  {row.airline:<7} {stops_label(row.transfers):<8} dep ~{row.departure_at[11:16]}  "
              f"₹{row.predicted_price:,.0f}")

    if live:
        cheapest = live[0]["price"]
        best_pred = predictions["predicted_price"].min()
        diff = (cheapest - best_pred) / best_pred
        verdict = "below" if diff < 0 else "above"
        print(f"\nCheapest live fare ₹{cheapest:,.0f} is {abs(diff):.0%} {verdict} the model's cheapest estimate.")


if __name__ == "__main__":
    main()
