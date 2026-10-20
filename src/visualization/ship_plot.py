
"""Visualize initial and final ship-loading states."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_experiments import DEFAULT_DATASET_DIR, load_problem

OUTPUT_DIR = PROJECT_ROOT / "output"


def draw_state(ax, state_data, ship, vehicles, title):
    """Draw one ship-loading state."""
    ax.set_xlim(0, ship.width)
    ax.set_ylim(0, ship.length)
    ax.set_xticks(range(ship.width + 1))
    ax.set_yticks(range(ship.length + 1))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("Ship width (x)")
    ax.set_ylabel("Ship length (y)")
    ax.set_title(title)

    # Outline of the ship deck.
    ax.add_patch(
        Rectangle(
            (0, 0),
            ship.width,
            ship.length,
            fill=False,
            linewidth=2,
        )
    )

    placements = state_data.get("placements", {})

    for vehicle_id, placement in placements.items():
        if placement is None:
            continue

        vehicle = vehicles[vehicle_id]
        x = placement["x"]
        y = placement["y"]
        orientation = placement["orientation"]

        # Horizontal keeps the original dimensions.
        # Vertical swaps the width and length.
        if orientation == "horizontal":
            width, length = vehicle.width, vehicle.length
        else:
            width, length = vehicle.length, vehicle.width

        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                length,
                alpha=0.65,
                edgecolor="black",
                linewidth=1.2,
            )
        )

        ax.text(
            x + width / 2,
            y + length / 2,
            vehicle_id,
            ha="center",
            va="center",
            fontsize=9,
        )


def plot_ship_states(dataset_dir: Path = DEFAULT_DATASET_DIR):
    """Create initial-versus-final plots using the selected CSV dataset."""
    states_path = OUTPUT_DIR / "states.json"

    if not states_path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {states_path}. "
            "Jalankan eksperimen terlebih dahulu."
        )

    with states_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not records:
        raise ValueError("states.json tidak berisi data.")

    ship, vehicles = load_problem(Path(dataset_dir).resolve())

    for record in records:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        draw_state(
            axes[0],
            record["initial_state"],
            ship,
            vehicles,
            f"Initial State\n{record['algorithm']}",
        )

        draw_state(
            axes[1],
            record["final_state"],
            ship,
            vehicles,
            f"Final State\n{record['algorithm']}",
        )

        fig.suptitle(
            f"Trial {record['trial']} | "
            f"Initial score: {record['initial_score']:.1f} | "
            f"Final score: {record['final_score']:.1f}"
        )
        fig.tight_layout()

        algorithm_slug = record["algorithm"].lower().replace(" ", "_")
        output_path = OUTPUT_DIR / (
            f"ship_states_trial_{record['trial']}_{algorithm_slug}.png"
        )

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Visualisasi state disimpan: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Folder berisi ship.csv dan vehicles.csv.",
    )
    args = parser.parse_args()
    plot_ship_states(args.dataset_dir)
