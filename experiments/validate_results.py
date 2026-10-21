
"""Validate every solution produced by HC and SA."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_DIR, OUTPUT_DIR
from src.core.dataset_loader import load_problem
from src.core.models import Orientation, Placement, State
from src.core.objective import calculate_objective
from src.core.constraints import is_valid_state


def deserialize_state(data):
    """Convert JSON back into State object."""
    placements = {}

    for vehicle_id, placement in data["placements"].items():
        if placement is None:
            placements[vehicle_id] = None
        else:
            placements[vehicle_id] = Placement(
                x=placement["x"],
                y=placement["y"],
                orientation=Orientation(placement["orientation"]),
            )

    return State(placements=placements)


def validate():
    ship, vehicles = load_problem(DATASET_DIR)

    states_path = OUTPUT_DIR / "states.json"

    if not states_path.exists():
        raise FileNotFoundError("states.json belum ada.")

    with states_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    print("=" * 70)
    print("VALIDASI HASIL EKSPERIMEN")
    print("=" * 70)

    total = 0
    valid = 0

    for record in records:
        total += 1

        best_state = deserialize_state(record["best_state"])

        constraint_ok = is_valid_state(best_state, vehicles, ship)
        score = calculate_objective(best_state, vehicles)
        score_ok = score == record["best_score"]

        status = "VALID" if constraint_ok and score_ok else "INVALID"

        if status == "VALID":
            valid += 1

        print(
            f"Trial {record['trial']} | "
            f"{record['algorithm']:<20} | "
            f"{status}"
        )

        if not constraint_ok:
            print("  -> Constraint violation")

        if not score_ok:
            print(
                f"  -> Score mismatch "
                f"({score} != {record['best_score']})"
            )

    print("-" * 70)
    print(f"Lolos validasi : {valid}/{total}")
    print(f"Gagal validasi : {total-valid}/{total}")

    if valid == total:
        print("SEMUA SOLUSI VALID")
    else:
        print("MASIH ADA SOLUSI YANG PERLU DIPERIKSA")


if __name__ == "__main__":
    validate()
