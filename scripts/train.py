"""Train the fare model on every fare collected in data/fares.db."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Windows consoles/pipes default to cp1252, which can't print the rupee sign.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from flightfare.model import train  # noqa: E402

if __name__ == "__main__":
    try:
        train()
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")
