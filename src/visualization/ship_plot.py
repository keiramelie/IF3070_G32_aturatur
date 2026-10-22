"""Visualize initial and final/best ship-loading states for HC, SA, and GA."""

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_DIR as DEFAULT_DATASET_DIR, OUTPUT_DIR
from experiments.ga_experiments import calculate_dataset_fingerprint
from src.core.constraints import is_valid_state
from src.core.dataset_loader import load_problem
from src.core.models import Vehicle, Ship, Placement, Orientation, State

BLUE_PALETTE = (
    "#D4EAF7", "#C6DFF0", "#A9CBE9", "#BBD9F3", "#D1E4F8",
    "#9FC4E5", "#E1F1FC", "#CAE4FA", "#D6EAF9",
)

def vehicle_blue(vehicle_id, vehicles):
    """Keep each vehicle's color consistent across initial/final states."""
    return BLUE_PALETTE[sorted(vehicles).index(vehicle_id) % len(BLUE_PALETTE)]

def draw_rounded_vehicle(ax, x, y, width, length, vehicle_id, vehicles):
    """Draw a rounded inset while preserving the exact grid footprint."""
    inset = min(0.10, width / 6, length / 6)
    rounded = FancyBboxPatch(
        (x + inset, y + inset),
        width - 2 * inset,
        length - 2 * inset,
        boxstyle="round,pad=0,rounding_size=0.16",
        linewidth=1.05,
        edgecolor="#1E3A5F",
        facecolor=vehicle_blue(vehicle_id, vehicles),
        zorder=2,
    )
    ax.add_patch(rounded)
    ax.text(
        x + width / 2,
        y + length / 2,
        vehicle_id,
        ha="center",
        va="center",
        fontsize=7 if min(width, length) <= 2 else 8,
        color="#0B2545",
        fontweight="semibold",
        zorder=3,
        clip_on=True,
    )

def state_placements(state_data):
    """Accept HC/SA's {placements: ...} and GA's flat placement mapping."""
    if not isinstance(state_data, dict):
        raise ValueError("State must be a placement mapping.")
    placements = state_data.get("placements", state_data)
    if not isinstance(placements, dict):
        raise ValueError("State placements must be a mapping.")
    return placements


def unloaded_vehicle_report(state_data, vehicles):
    """Text report of vehicles NOT loaded onto the ship (reusable in the UI)."""
    placements = state_placements(state_data)
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

    placements = state_placements(state_data)
    occupied = {}
    overlaps = set()
    loaded_ids, total_weight = [], 0

    try:
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
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed state in visualization: {title}") from error

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

        draw_rounded_vehicle(ax, x, y, width, length, vehicle_id, vehicles)

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


def plot_state_comparison(record, ship, vehicles, *, result_state="final"):
    """Use the same layout for local-search records and GA experiment runs."""
    if result_state not in ("final", "best"):
        raise ValueError("result_state must be 'final' or 'best'.")
    algorithm = record.get("algorithm", "Genetic Algorithm")
    result_label = result_state.capitalize()
    score = record[f"{result_state}_score"]

    fig = plt.figure(figsize=(17, 10))
    try:
        grid = fig.add_gridspec(1, 4, width_ratios=[1, 0.8, 1, 0.8], wspace=0.12)
        ax_init = fig.add_subplot(grid[0])
        ax_init_info = fig.add_subplot(grid[1])
        ax_result = fig.add_subplot(grid[2])
        ax_result_info = fig.add_subplot(grid[3])

        init_info = draw_state(
            ax_init, record["initial_state"], ship, vehicles,
            f"Initial State\n{algorithm}",
        )
        draw_info(ax_init_info, init_info, ship, vehicles, record["initial_score"])
        result_info = draw_state(
            ax_result, record[f"{result_state}_state"], ship, vehicles,
            f"{result_label} State\n{algorithm}",
        )
        draw_info(ax_result_info, result_info, ship, vehicles, score)

        details = [f"Trial {record['trial']}"]
        if "population_size" in record:
            details.extend([
                f"Seed {record['seed']}",
                f"Population {record['population_size']}",
                f"Generations {record['max_generations']}",
                record["study"].replace("_", " "),
            ])
        details.extend([
            f"Initial score: {record['initial_score']:.1f}",
            f"{result_label} score: {score:.1f}",
        ])
        fig.suptitle(" | ".join(details))
    except Exception:
        plt.close(fig)
        raise
    return fig


def plot_ship_states(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    *,
    states_path: Path | None = None,
    output_dir: Path | None = None,
    result_state: str = "final",
    study: str | None = None,
    population_size: int | None = None,
    generations: int | None = None,
    trial: int | None = None,
):
    """Read states.json or ga_runs.json and export one comparison per run.

    GA filenames include the run ID, so studies/settings never overwrite one
    another. The default still renders HC/SA's output/states.json as before.
    """
    if result_state not in ("final", "best"):
        raise ValueError("result_state must be 'final' or 'best'.")
    states_path = Path(states_path) if states_path is not None else OUTPUT_DIR / "states.json"
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR

    if not states_path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {states_path}. "
            "Jalankan eksperimen terlebih dahulu."
        )

    with states_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError(f"{states_path.name} harus berisi daftar run yang tidak kosong.")

    filters = {
        "study": study,
        "population_size": population_size,
        "max_generations": generations,
        "trial": trial,
    }
    records = [
        record for record in records
        if all(value is None or record.get(key) == value for key, value in filters.items())
    ]
    if not records:
        raise ValueError("Tidak ada run yang sesuai dengan filter visualisasi.")

    ship, vehicles = load_problem(Path(dataset_dir).resolve())
    # Equal dimensions/IDs alone cannot detect changed fees or vehicle weights.
    if any("dataset_sha256" in record for record in records):
        fingerprint = calculate_dataset_fingerprint(vehicles, ship)
        if any(record.get("dataset_sha256", fingerprint) != fingerprint for record in records):
            raise ValueError("Dataset CSV tidak cocok dengan dataset hasil eksperimen GA.")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []
    generated = []
    for record in records:
        algorithm = record.get("algorithm", "Genetic Algorithm")
        run_id = record.get("run_id")
        identifier = run_id or f"trial_{record['trial']}_{algorithm.lower().replace(' ', '_')}"
        identifier = re.sub(r"[^a-zA-Z0-9_()-]", "_", identifier)
        suffix = "_best" if result_state == "best" else ""
        output_path = output_dir / f"ship_states_{identifier}{suffix}.png"
        if output_path in generated:
            raise ValueError(f"Duplicate run would overwrite visualization: {identifier}")
        report_lines.append(
            f"=== Trial {record['trial']} | {algorithm} "
            f"{f'| {run_id} ' if run_id else ''}({result_state} state) ==="
        )
        report_lines.append(unloaded_vehicle_report(record[f"{result_state}_state"], vehicles))
        report_lines.append("")

        fig = plot_state_comparison(record, ship, vehicles, result_state=result_state)
        try:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
        generated.append(output_path)

        print(f"Visualisasi state disimpan: {output_path}")

    report_path = output_dir / f"unloaded_vehicles{suffix}.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Laporan kendaraan tidak dimuat disimpan: {report_path}")
    return generated


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

    for index, vehicle_id in enumerate(sorted(vehicles)):
        placement = state.placements[vehicle_id]

        if placement is None:
            continue

        vehicle = vehicles[vehicle_id]
        width, length = placement.dimension(vehicle)

        draw_rounded_vehicle(
            ax, placement.x, placement.y, width, length, vehicle_id, vehicles
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
    parser.add_argument(
        "--states", "--ga-runs", dest="states_path", type=Path,
        help="Path states.json (HC/SA) atau ga_runs.json (GA). Default: output/states.json.",
    )
    parser.add_argument("--output-dir", type=Path, help="Folder penyimpanan gambar dan laporan.")
    parser.add_argument("--result-state", choices=("final", "best"), default="final")
    parser.add_argument("--study", choices=("vary_generations", "vary_population"))
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--trial", type=int)
    args = parser.parse_args()
    plot_ship_states(
        args.dataset_dir,
        states_path=args.states_path,
        output_dir=args.output_dir,
        result_state=args.result_state,
        study=args.study,
        population_size=args.population_size,
        generations=args.generations,
        trial=args.trial,
    )
