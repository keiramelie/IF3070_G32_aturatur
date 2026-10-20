
"""Compare HC and SA experiment scores and runtimes."""

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


def load_summary(summary_path: Path):
    """Read and validate summary.csv."""
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"File summary tidak ditemukan: {summary_path}"
        )

    required_columns = {
        "trial",
        "algorithm",
        "best_score",
        "runtime_seconds",
    }

    records = []

    with summary_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("summary.csv tidak memiliki header.")

        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Kolom summary.csv tidak lengkap: {sorted(missing)}"
            )

        for row in reader:
            if row["algorithm"] not in ALGORITHMS:
                continue

            try:
                row["trial"] = int(row["trial"])
                row["best_score"] = float(row["best_score"])
                row["runtime_seconds"] = float(
                    row["runtime_seconds"]
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Data summary tidak valid: {row}"
                ) from error

            records.append(row)

    if not records:
        raise ValueError("Tidak ada data HC atau SA dalam summary.csv.")

    return records


def plot_trial_comparison(records, output_dir: Path, metric: str):
    """Compare one metric between algorithms for each trial."""
    labels = {
        "best_score": ("Best Score Comparison", "Total Shipping Fee"),
        "runtime_seconds": ("Runtime Comparison", "Runtime (seconds)"),
    }

    title, ylabel = labels[metric]
    trials = sorted({row["trial"] for row in records})

    x_positions = list(range(len(trials)))
    present = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in records)]
    bar_width = 0.8 / max(len(present), 1)

    fig, ax = plt.subplots(figsize=(11, 6))

    for index, algorithm in enumerate(present):
        values = []

        for trial in trials:
            matches = [
                row[metric]
                for row in records
                if row["trial"] == trial
                and row["algorithm"] == algorithm
            ]

            values.append(matches[0] if matches else 0)

        positions = [
            x + (index - (len(present) - 1) / 2) * bar_width
            for x in x_positions
        ]

        bars = ax.bar(
            positions,
            values,
            bar_width,
            label=algorithm,
        )
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=6)

    ax.set_title(title)
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(trial) for trial in trials])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    output_path = output_dir / f"{metric}_by_trial.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafik tersimpan: {output_path}")


def plot_average_comparison(records, output_dir: Path):
    """Compare mean best score and mean runtime."""
    grouped = defaultdict(list)

    for row in records:
        grouped[row["algorithm"]].append(row)

    averages = {}

    for algorithm in ALGORITHMS:
        rows = grouped.get(algorithm, [])

        if not rows:
            continue

        averages[algorithm] = {
            "best_score": (
                sum(row["best_score"] for row in rows) / len(rows)
            ),
            "runtime_seconds": (
                sum(row["runtime_seconds"] for row in rows)
                / len(rows)
            ),
        }

    # Average best score comparison.
    fig, ax = plt.subplots(figsize=(9, 6))

    algorithms = list(averages.keys())
    scores = [
        averages[algorithm]["best_score"]
        for algorithm in algorithms
    ]

    bars = ax.bar(algorithms, scores)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_title("Average Best Score Comparison")
    ax.set_ylabel("Average Total Shipping Fee")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    output_path = output_dir / "average_best_score.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafik tersimpan: {output_path}")

    # Average runtime comparison.
    fig, ax = plt.subplots(figsize=(9, 6))

    runtimes = [
        averages[algorithm]["runtime_seconds"]
        for algorithm in algorithms
    ]

    bars = ax.bar(algorithms, runtimes)
    ax.bar_label(bars, fmt="%.3f s", padding=3)
    ax.set_title("Average Runtime Comparison")
    ax.set_ylabel("Runtime (seconds)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    output_path = output_dir / "average_runtime.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafik tersimpan: {output_path}")

    # Print numeric summary for reporting.
    print("\nRINGKASAN RATA-RATA")
    for algorithm, values in averages.items():
        print(
            f"{algorithm}: "
            f"Best score = {values['best_score']:.2f}, "
            f"Runtime = {values['runtime_seconds']:.4f} detik"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare HC and SA experiment results."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(OUTPUT_DIR) / "summary.csv",
        help="Path ke summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help="Folder penyimpanan grafik.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_summary(args.summary)

    plot_trial_comparison(
        records, args.output_dir, "best_score"
    )
    plot_trial_comparison(
        records, args.output_dir, "runtime_seconds"
    )
    plot_average_comparison(records, args.output_dir)

    print("\nSemua visualisasi perbandingan selesai dibuat.")


if __name__ == "__main__":
    main()
