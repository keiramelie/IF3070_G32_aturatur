import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GA_RUNS = PROJECT_ROOT / "output/ga/synthetic_10x20_pilot/ga_runs.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/ga_visualization"


def plot_trial_progress(record: dict, destination: Path) -> None:
    """Plot population maximum, population average, and best-ever fitness."""
    maximum = record["maximum_fitness_history"]
    average = record["average_fitness_history"]
    generations = record["max_generations"]
    if len(maximum) != generations + 1 or len(average) != len(maximum):
        raise ValueError(f"Incomplete generation history: {record['run_id']}")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for values in (maximum, average)
        for value in values
    ):
        raise ValueError(f"Invalid fitness values: {record['run_id']}")
    if any(mean > best + 1e-8 for mean, best in zip(average, maximum)):
        raise ValueError(f"Average exceeds maximum fitness: {record['run_id']}")
    if not math.isclose(maximum[-1], record["final_score"]):
        raise ValueError(f"Final score disagrees with history: {record['run_id']}")
    best_score = max(maximum)
    best_generation = maximum.index(best_score)
    if not math.isclose(best_score, record["best_score"]):
        raise ValueError(f"Best score disagrees with history: {record['run_id']}")
    if record.get("best_generation", best_generation) != best_generation:
        raise ValueError(f"Best generation disagrees with history: {record['run_id']}")

    best_so_far = []
    running_best = float("-inf")
    for score in maximum:
        running_best = max(running_best, score)
        best_so_far.append(running_best)

    xs = list(range(generations + 1))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    try:
        ax.plot(xs, maximum, label="Generation maximum", color="#2563EB", linewidth=1.8)
        ax.plot(xs, average, label="Generation average", color="#EA8A16", linewidth=1.8)
        ax.plot(
            xs, best_so_far, label="Best-so-far", color="#17804B",
            linewidth=2.1, linestyle="--",
        )
        ax.axvline(best_generation, color="#17804B", linewidth=0.9, alpha=0.35)
        ax.scatter([best_generation], [best_score], color="#17804B", s=28, zorder=4)
        ax.set_title(
            f"GA Optimization Progress - Trial {record['trial']}, Seed {record['seed']}\n"
            f"Population {record['population_size']} | Generations {generations}"
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Total Shipping Fee")
        if generations:
            ax.set_xlim(0, generations)
        ax.grid(alpha=0.22)
        ax.legend(loc="lower right")
        fig.text(
            0.5, 0.02,
            f"Initial state: {record['initial_score']:g}  |  "
            f"Final: {record['final_score']:g}  |  "
            f"Best-ever: {best_score:g} at generation {best_generation}",
            ha="center", fontsize=9, color="#444444",
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(destination, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)


def export_trial_progress(
    ga_runs: Path,
    output_dir: Path,
    *,
    study: str = "vary_generations",
    population_size: int = 40,
    generations: int = 100,
    trial: int | None = None,
) -> list[Path]:
    with Path(ga_runs).open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("ga_runs.json must contain a nonempty list of GA runs.")
    selected = [
        record for record in records
        if record["study"] == study
        and record["population_size"] == population_size
        and record["max_generations"] == generations
        and (trial is None or record["trial"] == trial)
    ]
    if not selected:
        raise ValueError("No GA runs match the selected study, configuration, and trial.")
    run_ids = [record["run_id"] for record in selected]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate GA run IDs.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for record in sorted(selected, key=lambda row: row["trial"]):
        run_id = re.sub(r"[^a-zA-Z0-9_-]", "_", record["run_id"])
        destination = output_dir / f"ga_progress_{run_id}.png"
        if destination in generated:
            raise ValueError("Run IDs would overwrite the same progress figure.")
        plot_trial_progress(record, destination)
        generated.append(destination)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ga-runs", type=Path, default=DEFAULT_GA_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--study", choices=("vary_generations", "vary_population"), default="vary_generations")
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--trial", type=int)
    args = parser.parse_args()
    paths = export_trial_progress(
        args.ga_runs, args.output_dir, study=args.study,
        population_size=args.population_size, generations=args.generations,
        trial=args.trial,
    )
    for path in paths:
        print(f"Optimization progress disimpan: {path}")


if __name__ == "__main__":
    main()
