import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRIMARY_ALGORITHMS = ("Hill Climbing", "Simulated Annealing", "Genetic Algorithm")


def read_csv (path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json (path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def mean_and_sd (values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty group")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_ga (rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["study"] not in ("vary_generations", "vary_population"):
            raise ValueError("Unexpected GA study")
        grouped[(row["study"], int(row["setting"]))].append(row)
    summaries = []
    for (study, setting), group in sorted(grouped.items()):
        fitness, fitness_sd = mean_and_sd([float(row["best_score"]) for row in group])
        runtime, runtime_sd = mean_and_sd([float(row["runtime_seconds"]) for row in group])
        final, _ = mean_and_sd([float(row["final_score"]) for row in group])
        summaries.append({
            "study": study,
            "setting": setting,
            "trials": len(group),
            "mean_best_fitness": fitness,
            "sd_best_fitness": fitness_sd,
            "mean_final_fitness": final,
            "mean_runtime_seconds": runtime,
            "sd_runtime_seconds": runtime_sd,
            "population_size": int(group[0]["population_size"]),
            "max_generations": int(group[0]["max_generations"]),
        })
    return summaries


def validate_ga (
    rows: list[dict],
    history_rows: list[dict],
    runs: list[dict],
    manifest: dict,
) -> dict[str, list[dict]]:
    if len(rows) != 18 or len(runs) != 18 or manifest.get("number_of_runs") != 18:
        raise ValueError("GA input must contain all 18 recorded runs")
    seeds = set(manifest["seeds"])
    expected_studies = ("vary_generations", "vary_population")
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 18 or run_ids != {row["run_id"] for row in runs}:
        raise ValueError("GA run IDs are missing or duplicated")
    if any(row["dataset_id"] != manifest["dataset_id"] or row["dataset_sha256"] != manifest["dataset_sha256"] for row in rows):
        raise ValueError("GA dataset metadata does not agree with manifest")
    if any(row["study"] not in expected_studies for row in rows):
        raise ValueError("Unexpected GA study")
    for study in expected_studies:
        group = [row for row in rows if row["study"] == study]
        settings = {int(row["setting"]) for row in group}
        if len(group) != 9 or len(settings) != 3:
            raise ValueError(f"Expected nine GA runs and three settings in {study}")
        for setting in settings:
            matching = [row for row in group if int(row["setting"]) == setting]
            if {int(row["seed"]) for row in matching} != seeds or len(matching) != 3:
                raise ValueError(f"Missing or repeated GA trials for {study}:{setting}")
    by_run = defaultdict(list)
    for row in history_rows:
        by_run[row["run_id"]].append(row)
    for row in rows:
        history = sorted(by_run.get(row["run_id"], []), key=lambda item: int(item["generation"]))
        length = int(row["max_generations"]) + 1
        if len(history) != length or [int(item["generation"]) for item in history] != list(range(length)):
            raise ValueError(f"Invalid GA generation history: {row['run_id']}")
        maximum = [float(item["maximum_fitness"]) for item in history]
        average = [float(item["average_fitness"]) for item in history]
        if any(avg > best + 1e-8 for avg, best in zip(average, maximum)):
            raise ValueError(f"Average exceeds maximum fitness: {row['run_id']}")
        if abs(max(maximum) - float(row["best_score"])) > 1e-8:
            raise ValueError(f"GA best-ever fitness disagrees with history: {row['run_id']}")
        if abs(maximum[-1] - float(row["final_score"])) > 1e-8:
            raise ValueError(f"GA final fitness disagrees with history: {row['run_id']}")
        if int(row["fitness_evaluations"]) != int(row["population_size"]) * length:
            raise ValueError(f"Incorrect GA fitness evaluation count: {row['run_id']}")
    if set(by_run) != run_ids:
        raise ValueError("GA history contains unexpected run IDs")
    return by_run


def select_ga_configuration (rows: list[dict], population_size: int, generations: int) -> list[dict]:
    matches = [
        row for row in rows
        if int(row["population_size"]) == population_size
        and int(row["max_generations"]) == generations
    ]
    by_seed = defaultdict(list)
    for row in matches:
        by_seed[int(row["seed"])].append(row)
    if len(by_seed) != 3:
        raise ValueError("Requested GA configuration must have three distinct seed trials")
    selected = []
    for seed, group in sorted(by_seed.items()):
        first = group[0]
        if any(
            any(row[key] != first[key] for key in ("initial_score", "final_score", "best_score"))
            for row in group[1:]
        ):
            raise ValueError(f"Duplicate configuration disagrees for seed {seed}")
        selected.append(next((row for row in group if row["study"] == "vary_generations"), first))
    return selected


def validate_local (
    rows: list[dict],
    states: list[dict],
    ga_selected: list[dict],
    ga_runs: list[dict],
    dataset_id: str,
) -> dict[str, list[dict]]:
    if any(row["dataset_id"] != dataset_id or row["objective"] != "fee" for row in rows):
        raise ValueError("HC/SA results must match GA dataset and mandatory fee objective")
    if any(record["dataset_id"] != dataset_id for record in states):
        raise ValueError("HC/SA state records do not match GA dataset")
    group = defaultdict(list)
    for row in rows:
        group[row["algorithm"]].append(row)
    seeds = {int(row["seed"]) for row in ga_selected}
    for algorithm in ("Hill Climbing", "Simulated Annealing"):
        matched = group.get(algorithm, [])
        if len(matched) != 3 or {int(row["seed"]) for row in matched} != seeds:
            raise ValueError(f"HC/SA seed mismatch or missing trials for {algorithm}")
    runs_by_id = {row["run_id"]: row for row in ga_runs}
    local_states = {(row["algorithm"], int(row["seed"])): row for row in states}
    for ga in ga_selected:
        seed = int(ga["seed"])
        ga_initial = runs_by_id[ga["run_id"]]["initial_state"]
        for algorithm in ("Hill Climbing", "Simulated Annealing"):
            reference = local_states.get((algorithm, seed))
            if reference is None or reference["initial_state"]["placements"] != ga_initial:
                raise ValueError(f"GA and {algorithm} initial states differ for seed {seed}")
            local_summary = next(row for row in group[algorithm] if int(row["seed"]) == seed)
            if abs(float(local_summary["initial_score"]) - float(ga["initial_score"])) > 1e-8:
                raise ValueError(f"GA and {algorithm} initial scores differ for seed {seed}")
    return group


def write_csv (path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to export")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_parameter_effect (summaries: list[dict], study: str, destination: Path) -> None:
    rows = sorted((row for row in summaries if row["study"] == study), key=lambda row: row["setting"])
    if len(rows) != 3:
        raise ValueError(f"Missing GA study settings for {study}")
    xs = [row["setting"] for row in rows]
    ys = [row["mean_best_fitness"] for row in rows]
    errors = [row["sd_best_fitness"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, ys, yerr=errors, marker="o", capsize=5)
    ax.set_xticks(xs)
    ax.set_xlabel("Generations" if study == "vary_generations" else "Population size")
    ax.set_ylabel("Mean best-ever ShippingFee")
    ax.set_title("GA: effect of generations" if study == "vary_generations" else "GA: effect of population size")
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.14, 0.02, "Error bars: sample standard deviation, three seeded trials per setting", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_convergence (
    selected: list[dict],
    histories: dict[str, list[dict]],
    population_size: int,
    generations: int,
    destination: Path,
) -> None:
    curves = []
    for row in selected:
        history = sorted(histories[row["run_id"]], key=lambda item: int(item["generation"]))
        maximum = [float(item["maximum_fitness"]) for item in history]
        average = [float(item["average_fitness"]) for item in history]
        best_so_far = []
        best = float("-inf")
        for value in maximum:
            best = max(best, value)
            best_so_far.append(best)
        curves.append((maximum, average, best_so_far))
    xs = list(range(generations + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    for position, label in enumerate(("Current generation maximum", "Current generation average", "Best-so-far")):
        ys = [statistics.mean(curve[position][generation] for curve in curves) for generation in xs]
        ax.plot(xs, ys, label=label)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean ShippingFee over three trials")
    ax.set_title(f"GA convergence: population {population_size}, generations {generations}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def summarize_comparison (selected: list[dict], local_groups: dict[str, list[dict]]) -> list[dict]:
    groups = {"Genetic Algorithm": selected, **local_groups}
    summaries = []
    for algorithm, rows in groups.items():
        best_mean, best_sd = mean_and_sd([float(row["best_score"]) for row in rows])
        time_mean, time_sd = mean_and_sd([float(row["runtime_seconds"]) for row in rows])
        summaries.append({
            "algorithm": algorithm,
            "trials": len(rows),
            "mean_best_fitness": best_mean,
            "sd_best_fitness": best_sd,
            "mean_runtime_seconds": time_mean,
            "sd_runtime_seconds": time_sd,
        })
    return sorted(summaries, key=lambda row: (
        PRIMARY_ALGORITHMS.index(row["algorithm"])
        if row["algorithm"] in PRIMARY_ALGORITHMS else len(PRIMARY_ALGORITHMS),
        row["algorithm"],
    ))


def plot_algorithm_comparison (rows: list[dict], metric: str, destination: Path, include_bonus: bool) -> None:
    selected = [row for row in rows if include_bonus or row["algorithm"] in PRIMARY_ALGORITHMS]
    if not selected:
        raise ValueError("No algorithms available for comparison")
    labels = [row["algorithm"] for row in selected]
    means = [row[f"mean_{metric}"] for row in selected]
    errors = [row[f"sd_{metric}"] for row in selected]
    fig, ax = plt.subplots(figsize=(9, max(4.8, len(selected) * 0.8)))
    positions = list(range(len(selected)))
    if metric == "best_fitness":
        ax.errorbar(means, positions, xerr=errors, fmt="o", capsize=5)
        lower = min(mean - error for mean, error in zip(means, errors))
        upper = max(mean + error for mean, error in zip(means, errors))
        padding = max((upper - lower) * 0.15, 15)
        ax.set_xlim(lower - padding, upper + padding)
    else:
        ax.barh(positions, means, xerr=errors, capsize=4)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Best-ever ShippingFee" if metric == "best_fitness" else "Recorded runtime (seconds)")
    ax.set_title("Algorithm solution quality" if metric == "best_fitness" else "Recorded algorithm runtime")
    ax.grid(axis="x", alpha=0.2)
    if metric == "runtime_seconds":
        fig.text(0.1, 0.02, "Runtime is descriptive; do not infer speed rankings across different hardware.", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def file_hash (path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_analysis (
    ga_dir: Path,
    output_dir: Path,
    *,
    population_size: int,
    generations: int,
    local_dir: Path | None = None,
    include_bonus: bool = False,
) -> list[Path]:
    ga_dir = ga_dir.resolve()
    output_dir = output_dir.resolve()
    local_dir = local_dir.resolve() if local_dir is not None else None
    if output_dir in (ga_dir, local_dir):
        raise ValueError("Analysis output directory must differ from input directories")
    ga_paths = {name: ga_dir / name for name in (
        "ga_manifest.json", "ga_runs.json", "ga_summary.csv", "ga_history.csv"
    )}
    ga_manifest = read_json(ga_paths["ga_manifest.json"])
    ga_runs = read_json(ga_paths["ga_runs.json"])
    ga_summary = read_csv(ga_paths["ga_summary.csv"])
    ga_history = read_csv(ga_paths["ga_history.csv"])
    histories = validate_ga(ga_summary, ga_history, ga_runs, ga_manifest)
    selected = select_ga_configuration(ga_summary, population_size, generations)
    selected_ids = {row["run_id"] for row in selected}
    if any(len(histories[run_id]) != generations + 1 for run_id in selected_ids):
        raise ValueError("Selected GA histories disagree with generation count")
    ga_summaries = summarize_ga(ga_summary)
    local_paths = {}
    local_groups = {}
    if local_dir is not None:
        local_paths = {name: local_dir / name for name in ("summary.csv", "states.json")}
        local_rows = read_csv(local_paths["summary.csv"])
        local_states = read_json(local_paths["states.json"])
        local_groups = validate_local(local_rows, local_states, selected, ga_runs, ga_manifest["dataset_id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    write_csv(output_dir / "ga_statistics.csv", ga_summaries)
    generated.append(output_dir / "ga_statistics.csv")
    for study, filename in (
        ("vary_generations", "ga_fitness_vs_generations.png"),
        ("vary_population", "ga_fitness_vs_population.png"),
    ):
        path = output_dir / filename
        plot_parameter_effect(ga_summaries, study, path)
        generated.append(path)
    path = output_dir / f"ga_convergence_pop{population_size}_gen{generations}.png"
    plot_convergence(selected, histories, population_size, generations, path)
    generated.append(path)
    if local_groups:
        comparison = summarize_comparison(selected, local_groups)
        path = output_dir / "algorithm_comparison.csv"
        write_csv(path, comparison)
        generated.append(path)
        for metric, name in (
            ("best_fitness", "algorithm_best_fitness.png"),
            ("runtime_seconds", "algorithm_runtime.png"),
        ):
            path = output_dir / name
            plot_algorithm_comparison(comparison, metric, path, include_bonus)
            generated.append(path)
    inputs = {name: file_hash(path) for name, path in {**ga_paths, **{f"local_{name}": path for name, path in local_paths.items()}}.items()}
    manifest = {
        "dataset_id": ga_manifest["dataset_id"],
        "dataset_sha256": ga_manifest["dataset_sha256"],
        "comparison_population_size": population_size,
        "comparison_generations": generations,
        "seeds": ga_manifest["seeds"],
        "include_bonus": include_bonus,
        "input_sha256": inputs,
        "generated_files": [path.name for path in generated],
    }
    path = output_dir / "analysis_manifest.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    generated.append(path)
    return generated


def main () -> None:
    parser = argparse.ArgumentParser(description="Create reproducible GA and HC/SA experiment figures")
    parser.add_argument("--ga-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--include-bonus", action="store_true")
    args = parser.parse_args()
    generated = run_analysis(
        args.ga_dir,
        args.output_dir,
        population_size=args.population_size,
        generations=args.generations,
        local_dir=args.local_dir,
        include_bonus=args.include_bonus,
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
