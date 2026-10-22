"""Visualize results from Hill Climbing and Simulated Annealing."""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"


def read_csv(path):
    """Read CSV records and validate that the file contains data."""
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}. "
            "Jalankan experiments/run_experiments.py terlebih dahulu."
        )

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Header CSV tidak ditemukan di {path.name}.")

        rows = list(reader)

    if not rows:
        raise ValueError(f"Tidak ada data di dalam {path.name}.")

    return rows


def save_plot(filename):
    """Save the current Matplotlib figure to the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik disimpan: {output_path}")


def plot_sa_exponential_history():
    """Plot exp(delta E / temperature) for Simulated Annealing."""
    rows = read_csv(OUTPUT_DIR / "sa_metrics.csv")
    histories = defaultdict(list)

    for row in rows:
        trial = row.get("trial")
        iteration = row.get("iteration")
        value = row.get("exp_delta_e_over_t")

        if not all((trial, iteration, value is not None)):
            continue

        value = float(value)
        if math.isfinite(value):
            histories[trial].append((int(iteration), value))

    if not histories:
        raise ValueError(
            "sa_metrics.csv tidak memiliki nilai valid "
            "untuk exp_delta_e_over_t."
        )

    plt.figure(figsize=(10, 6))

    for trial, values in sorted(histories.items()):
        values.sort(key=lambda item: item[0])
        iterations = [item[0] for item in values]
        exponential_values = [item[1] for item in values]

        plt.plot(
            iterations,
            exponential_values,
            label=f"Trial {trial}",
            linewidth=0.8,
        )

    # e^(dE/T) spans many orders of magnitude (dE>0 at low T explodes),
    # so use a log axis; y=1 means dE=0 (neutral move).
    plt.yscale("log")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title(r"Simulated Annealing: $e^{\Delta E/T}$ per Iteration (log scale)")
    plt.xlabel("Iteration")
    plt.ylabel(r"$e^{\Delta E/T}$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("sa_exponential_history.png")


def plot_sa_acceptance_probability():
    """Plot acceptance probability across SA iterations."""
    rows = read_csv(OUTPUT_DIR / "sa_metrics.csv")

    if "acceptance_probability" not in rows[0]:
        print(
            "Grafik probabilitas penerimaan SA dilewati: "
            "kolom acceptance_probability tidak ditemukan. "
            "Jalankan ulang experiments/run_experiments.py "
            "dengan kode terbaru."
        )
        return

    histories = defaultdict(list)

    for row in rows:
        trial = row.get("trial")
        iteration = row.get("iteration")
        probability = row.get("acceptance_probability")

        if not all((trial, iteration, probability is not None)):
            continue

        probability = float(probability)
        if math.isfinite(probability):
            histories[trial].append((int(iteration), probability))

    if not histories:
        print(
            "Grafik probabilitas penerimaan SA dilewati: "
            "tidak ada data probabilitas yang valid."
        )
        return

    plt.figure(figsize=(10, 6))

    for trial, values in sorted(histories.items()):
        values.sort(key=lambda item: item[0])
        iterations = [item[0] for item in values]
        probabilities = [item[1] for item in values]

        plt.plot(
            iterations,
            probabilities,
            label=f"Trial {trial}",
        )

    plt.title("Simulated Annealing: Acceptance Probability")
    plt.xlabel("Iteration")
    plt.ylabel("Acceptance Probability")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("sa_acceptance_probability.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_sa_exponential_history()
    plot_sa_acceptance_probability()

    print("\nSemua grafik yang tersedia berhasil dibuat.")


if __name__ == "__main__":
    main()
