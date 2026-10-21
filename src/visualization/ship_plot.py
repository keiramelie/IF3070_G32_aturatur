"""Visualize initial and final ship-loading states."""

import argparse
import json
import sys
import textwrap
from pathlib import Path
from src.core.models import Vehicle, Ship, Placement, Orientation, State
from src.core.constraints import is_valid_state
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_experiments import DEFAULT_DATASET_DIR, load_problem
from src.core.models import Orientation, Placement

OUTPUT_DIR = PROJECT_ROOT / "output"


def unloaded_vehicle_report(state_data, vehicles):
    """Text report of vehicles NOT loaded onto the ship (reusable in the UI)."""
    placements = state_data.get("placements", {})
    unloaded = [vehicles[v] for v in sorted(vehicles) if placements.get(v) is None]
    lost_fee = sum(v.shipping_fee for v in unloaded)
    lines = [
        f"Kendaraan tidak dimuat: {len(unloaded)} dari {len(vehicles)} "
        f"(total fee tidak terangkut: {lost_fee:g})"
    ]
    for v in unloaded:
        lines.append(
            f"  {v.id}: ukuran {v.width}x{v.length}, "
            f"fee {v.shipping_fee:g}, berat {v.weight:g}, ETA {v.eta}"
        )
    if not unloaded:
        lines.append("  (semua kendaraan dimuat)")
    return "\n".join(lines)


def draw_state(ax, state_data, ship, vehicles, title):
    """Draw one ship-loading state.

    Dimensions come from ``Placement.dimension`` (the SAME rule the
    constraint checker uses: horizontal = (long, short), vertical =
    (short, long)). Drawing with the raw vehicle.width/length would show
    boxes that do not match the validated layout.
    The y axis points DOWN so (0, 0) is the top-left cell, as in
    ``build_grid`` (grid[y][x]).
    """
    ax.set_xlim(0, ship.width)
    ax.set_ylim(ship.length, 0)
    ax.set_xticks(range(ship.width + 1))
    ax.set_yticks(range(ship.length + 1))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Ship width (x)")
    ax.set_ylabel("Ship length (y)")
    ax.set_title(title)

    # Outline of the ship deck.
    ax.add_patch(
        Rectangle((0, 0), ship.width, ship.length, fill=False, linewidth=2)
    )

    placements = state_data.get("placements", {})
    palette = colormaps["tab20"]
    occupied = {}
    overlaps = set()
    loaded_ids, total_weight = [], 0

    state = State(
        placements={
            vehicle_id: (
                None if placement is None else Placement(
                    x=placement["x"],
                    y=placement["y"],
                    orientation=Orientation(placement["orientation"])
                )
            )
            for vehicle_id, placement in placements.items()
        }
    )

    if not is_valid_state(state, vehicles, ship):
        raise ValueError(f"Invalid state detected in visualization: {title}")

    for index, (vehicle_id, placement) in enumerate(sorted(placements.items())):
        if placement is None:
            continue

        vehicle = vehicles[vehicle_id]

        x = placement["x"]
        y = placement["y"]
        orientation = Orientation(placement["orientation"])

        placement_obj = Placement(x=x, y=y, orientation=orientation)

        width, length = placement_obj.dimension(vehicle)

        loaded_ids.append(vehicle_id)
        total_weight += vehicle.weight

        for cy in range(y, y + length):
            for cx in range(x, x + width):
                if (cx, cy) in occupied:
                    overlaps.add((occupied[(cx, cy)], vehicle_id))
                occupied[(cx, cy)] = vehicle_id

        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                length,
                facecolor=palette(index % 20),
                alpha=0.8,
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
            fontsize=8 if min(width, length) > 1 else 6,
            rotation=90 if length > width and width == 1 else 0,
        )

    unloaded = sorted(v for v in vehicles if placements.get(v) is None)
    return {
        "loaded": loaded_ids,
        "unloaded": unloaded,
        "weight": total_weight,
        "overlaps": sorted(overlaps),  # harus kosong untuk state valid
    }


def draw_info(ax, info, ship, vehicles, score):
    """Side panel (text only): value, weight, and vehicles left outside."""
    ax.axis("off")
    unloaded = info["unloaded"]
    lost_fee = sum(vehicles[v].shipping_fee for v in unloaded)
    y = 0.98

    def label(text):
        nonlocal y
        ax.text(0, y, text, transform=ax.transAxes, va="top",
                fontsize=9, color="#555555")
        y -= 0.035

    def value(text, size=11, bold=False, step=0.05):
        nonlocal y
        ax.text(0, y, text, transform=ax.transAxes, va="top",
                fontsize=size, fontweight="bold" if bold else "normal")
        y -= step

    label("Nilai")
    value(f"{score:g}", size=22, bold=True, step=0.065)
    value(f"Berat {info['weight']:g} dari {ship.max_capacity:g}", size=9, step=0.06)

    label("Dimuat")
    value(f"{len(info['loaded'])} dari {len(vehicles)} kendaraan", step=0.06)

    label(f"Di luar kapal ({len(unloaded)})")
    names = textwrap.fill(", ".join(unloaded) if unloaded else "-", 26)
    value(names, size=10, step=0.03 * (names.count("\n") + 1) + 0.04)

    label("Fee tidak terangkut")
    value(f"{lost_fee:g}", step=0.06)

    if info["overlaps"]:
        value(f"WARNING overlap: {info['overlaps']}", size=9)


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

    report_lines = []
    for record in records:
        report_lines.append(
            f"=== Trial {record['trial']} | {record['algorithm']} (final state) ==="
        )
        report_lines.append(unloaded_vehicle_report(record["final_state"], vehicles))
        report_lines.append("")

        fig = plt.figure(figsize=(17, 10))
        grid = fig.add_gridspec(1, 4, width_ratios=[1, 0.8, 1, 0.8], wspace=0.12)
        ax_init = fig.add_subplot(grid[0])
        ax_init_info = fig.add_subplot(grid[1])
        ax_final = fig.add_subplot(grid[2])
        ax_final_info = fig.add_subplot(grid[3])

        init_info = draw_state(
            ax_init, record["initial_state"], ship, vehicles,
            f"Initial State\n{record['algorithm']}",
        )
        draw_info(ax_init_info, init_info, ship, vehicles, record["initial_score"])

        final_info = draw_state(
            ax_final, record["final_state"], ship, vehicles,
            f"Final State\n{record['algorithm']}",
        )
        draw_info(ax_final_info, final_info, ship, vehicles, record["final_score"])

        fig.suptitle(
            f"Trial {record['trial']} | "
            f"Initial score: {record['initial_score']:.1f} | "
            f"Final score: {record['final_score']:.1f}"
        )

        algorithm_slug = record["algorithm"].lower().replace(" ", "_")
        output_path = OUTPUT_DIR / (
            f"ship_states_trial_{record['trial']}_{algorithm_slug}.png"
        )

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Visualisasi state disimpan: {output_path}")

    report_path = OUTPUT_DIR / "unloaded_vehicles.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Laporan kendaraan tidak dimuat disimpan: {report_path}")


def plot_ship(state: State, vehicles: dict[str, Vehicle], ship: Ship, title: str = "Ship Layout"):
    if not is_valid_state(state, vehicles, ship):
        raise ValueError("Cannot visualize an invalid state.")

    fig, ax = plt.subplots(
        figsize=(max(5, ship.width * 0.5),
                 max(5, ship.length * 0.35)))

    ax.set_xlim(0, ship.width)
    ax.set_ylim(ship.length, 0)

    ax.set_aspect("equal")
    ax.set_xticks(range(ship.width + 1))
    ax.set_yticks(range(ship.length + 1))
    ax.grid(True, alpha=0.25)

    ax.set_xlabel("Ship width (x)")
    ax.set_ylabel("Ship length (y)")
    ax.set_title(title, fontsize=13, fontweight="bold")

    palette = plt.get_cmap("tab20")

    for index, vehicle_id in enumerate(sorted(vehicles)):
        placement = state.placements[vehicle_id]

        if placement is None:
            continue

        vehicle = vehicles[vehicle_id]
        width, length = placement.dimension(vehicle)

        rectangle = Rectangle(
            (placement.x, placement.y),
            width,
            length,
            facecolor=palette(index % 20),
            edgecolor="#263238",
            linewidth=1,
            alpha=0.8,
            zorder=2,
        )

        ax.add_patch(rectangle)

        font_size = 7 if min(width, length) <= 2 else 9

        ax.text(
            placement.x + width / 2,
            placement.y + length / 2,
            vehicle_id,
            ha="center",
            va="center",
            fontsize=font_size,
            color="#17212b",
            zorder=3,
            clip_on=True,
        )

    fig.tight_layout()
    return fig

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