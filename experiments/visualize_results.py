
"""Visualize HC and SA convergence from experiment history."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from config import OUTPUT_DIR


ALGORITHMS = [
    "Hill Climbing",
    "Hill Climbing (Sideways)",
    "Hill Climbing (Stochastic)",
    "Hill Climbing (Random Restart)",
    "Simulated Annealing",
]


def load_history(history_path: Path):
    """Load and validate experiment history using built-in csv."""
    if not history_path.is_file():
        raise FileNotFoundError(
            f"File history tidak ditemukan: {history_path}"
        )

    required_columns = {
        "dataset_id",
        "trial",
        "seed",
        "algorithm",
        "iteration",
        "current_score",
    }

    records = []

    with history_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("history.csv kosong atau tidak memiliki header.")

        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Kolom history.csv tidak lengkap: {sorted(missing)}"
            )

        for row in reader:
            if row["algorithm"] not in ALGORITHMS:
                continue

            try:
                row["trial"] = int(row["trial"])
                row["iteration"] = int(row["iteration"])
                row["current_score"] = float(row["current_score"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Data history.csv tidak valid: {row}"
                ) from error

            records.append(row)

    if not records:
        raise ValueError("Tidak ada data HC atau SA di history.csv.")

    return records


def plot_convergence_by_trial(records, output_dir: Path):
    """Create one convergence plot for each trial."""
    trials = sorted({row["trial"] for row in records})

    for trial in trials:
        fig, ax = plt.subplots(figsize=(10, 6))

        for algorithm in ALGORITHMS:
            rows = [
                row for row in records
                if row["trial"] == trial
                and row["algorithm"] == algorithm
            ]
            rows.sort(key=lambda row: row["iteration"])

            if not rows:
                continue

            iterations = [row["iteration"] for row in rows]
            scores = [row["current_score"] for row in rows]

            ax.plot(
                iterations,
                scores,
                label=algorithm,
                marker="o" if len(rows) < 30 else None,
                markersize=3,
            )

        ax.set_title(f"Convergence Comparison - Trial {trial}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total Shipping Fee")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        path = output_dir / f"convergence_trial_{trial}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Grafik tersimpan: {path}")


def plot_average_convergence(records, output_dir: Path):
    """Plot mean score across trials at each iteration."""
    grouped_scores = defaultdict(list)

    for row in records:
        key = (row["algorithm"], row["iteration"])
        grouped_scores[key].append(row["current_score"])

    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm in ALGORITHMS:
        iterations = sorted(
            iteration
            for (name, iteration) in grouped_scores
            if name == algorithm
        )

        if not iterations:
            continue

        average_scores = [
            sum(grouped_scores[(algorithm, iteration)])
            / len(grouped_scores[(algorithm, iteration)])
            for iteration in iterations
        ]

        ax.plot(
            iterations,
            average_scores,
            label=algorithm,
        )

    ax.set_title("Average Convergence - HC vs SA")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Total Shipping Fee")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = output_dir / "convergence_average.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafik tersimpan: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize HC/SA convergence history."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path(OUTPUT_DIR) / "history.csv",
        help="Path ke file history.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help="Folder untuk menyimpan grafik.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_history(args.history)

    plot_convergence_by_trial(records, args.output_dir)
    plot_average_convergence(records, args.output_dir)

    print("\nSemua visualisasi konvergensi selesai dibuat.")


if __name__ == "__main__":
    main()
