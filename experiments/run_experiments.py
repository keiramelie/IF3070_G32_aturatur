
"""Run reproducible HC/SA experiments using configurable dataset input.

Default dataset directory and experiment parameters are read from config.py.

Example:
    py -3.14 -m experiments.run_experiments

For an alternative dataset:
    py -3.14 -m experiments.run_experiments --dataset-dir data/official
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path


# Ensure the project root is available for imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Algorithm and core imports.
from src.algorithms.hill_climbing import (
    run_hill_climbing,
    run_hill_climbing_random_restart,
    run_hill_climbing_sideways,
    run_hill_climbing_stochastic,
)
from src.algorithms.simulated_annealing import run_simulated_annealing
from src.core.initializer import generate_random_state
from src.core.models import State
from src.algorithms.objectives import OBJECTIVES
from src.core.objective import calculate_objective
from src.core.constraints import is_valid_state
from src.core.dataset_loader import load_problem

# Centralized project configuration.
from config import (
    DATASET_DIR,
    OUTPUT_DIR,
    TRIAL_SEEDS,
    LOAD_PROBABILITY,
    HC_MAX_ITERATIONS,
    SA_INITIAL_TEMPERATURE,
    SA_COOLING_RATE,
    SA_MIN_TEMPERATURE,
    SA_MAX_ITERATIONS,
    HC_MAX_SIDEWAYS,
    HC_MAX_RESTARTS,
    SA_STUCK_CHECK_INTERVAL,
)


DEFAULT_DATASET_DIR = DATASET_DIR


def serialize_value(value):
    """Convert model objects into JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: serialize_value(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            key: serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]

    return value


def serialize_state(state: State):
    """Serialize a state into JSON-compatible data."""
    return serialize_value(state)


def write_csv(path: Path, rows: list, fieldnames: list):
    """Write experiment records to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(dataset_dir: Path = DEFAULT_DATASET_DIR, bonus: bool = False,
                    objective_name: str = "fee"):
    """Run HC and SA experiments and save their results."""
    dataset_dir = Path(dataset_dir).resolve()

    # Load ship and vehicle data using the shared dataset loader.
    ship, vehicles = load_problem(dataset_dir)

    dataset_id = dataset_dir.name
    objective = OBJECTIVES[objective_name]
    # Objective wajib -> output/; objective bonus -> output/objective_<nama>/
    output_dir = Path(OUTPUT_DIR)
    if objective_name != "fee":
        output_dir = output_dir / f"objective_{objective_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    history_rows = []
    sa_metric_rows = []
    state_records = []

    print(f"Dataset: {dataset_id} ({len(vehicles)} kendaraan)")
    print(
        f"Kapal: {ship.width} x {ship.length}; "
        f"kapasitas: {ship.max_capacity}\n"
    )

    # Run every configured trial using a reproducible seed.
    for trial, seed in enumerate(TRIAL_SEEDS, start=1):
        initial_state = generate_random_state(
            vehicles=vehicles,
            ship=ship,
            load_prob=LOAD_PROBABILITY,
            seed=seed,
        )

        if not is_valid_state(initial_state, vehicles, ship):
            raise ValueError(
                f"Initial state tidak valid pada trial {trial}."
            )

        algorithms = [
            (
                "Hill Climbing",
                lambda: run_hill_climbing(
                    initial_state=initial_state,
                    vehicles=vehicles,
                    ship=ship,
                    max_iterations=HC_MAX_ITERATIONS,
                    objective=objective,
                ),
            ),
            (
                "Simulated Annealing",
                lambda: run_simulated_annealing(
                    initial_state=initial_state,
                    vehicles=vehicles,
                    ship=ship,
                    initial_temperature=SA_INITIAL_TEMPERATURE,
                    cooling_rate=SA_COOLING_RATE,
                    min_temperature=SA_MIN_TEMPERATURE,
                    max_iterations=SA_MAX_ITERATIONS,
                    seed=seed,
                    stuck_check_interval=SA_STUCK_CHECK_INTERVAL,
                    objective=objective,
                ),
            ),
        ]

        if bonus:
            algorithms += [
                ("Hill Climbing (Sideways)", lambda: run_hill_climbing_sideways(
                    initial_state, vehicles, ship,
                    max_sideways=HC_MAX_SIDEWAYS,
                    max_iterations=HC_MAX_ITERATIONS, seed=seed,
                    objective=objective)),
                ("Hill Climbing (Stochastic)", lambda: run_hill_climbing_stochastic(
                    initial_state, vehicles, ship,
                    max_iterations=HC_MAX_ITERATIONS, seed=seed,
                    objective=objective)),
                ("Hill Climbing (Random Restart)", lambda: run_hill_climbing_random_restart(
                    initial_state, vehicles, ship,
                    max_restarts=HC_MAX_RESTARTS,
                    max_iterations=HC_MAX_ITERATIONS,
                    load_prob=LOAD_PROBABILITY, seed=seed,
                    objective=objective)),
            ]

        for algorithm_name, run_algorithm in algorithms:
            result = run_algorithm()

            initial_score = objective(
                result.initial_state, vehicles
            )
            final_score = objective(
                result.final_state, vehicles
            )
            best_score = objective(
                result.best_state, vehicles
            )

            # Validate algorithm output and objective consistency.
            if best_score != result.best_score:
                raise ValueError(
                    f"{algorithm_name}: best_score tidak cocok "
                    "dengan best_state."
                )

            if not is_valid_state(result.final_state, vehicles, ship):
                raise ValueError(
                    f"{algorithm_name}: final_state tidak valid."
                )

            if not is_valid_state(result.best_state, vehicles, ship):
                raise ValueError(
                    f"{algorithm_name}: best_state tidak valid."
                )

            # Record one summary row for each algorithm and trial.
            summary_rows.append({
                "dataset_id": dataset_id,
                "trial": trial,
                "seed": seed,
                "algorithm": algorithm_name,
                "objective": objective_name,
                "initial_score": initial_score,
                "final_score": final_score,
                "best_score": best_score,
                "fee_score": calculate_objective(result.best_state, vehicles),
                "runtime_seconds": result.runtime,
                "iterations": result.iterations,
                "stuck_count": result.metadata.get(
                    "stuck_count", ""
                ),
                "stuck_episodes": result.metadata.get(
                    "stuck_episodes", ""
                ),
                "stuck_frequency": result.metadata.get(
                    "stuck_frequency", ""
                ),
                "stop_reason": result.metadata.get("stop_reason", ""),
                "sideways_moves": result.metadata.get("sideways_moves", ""),
                "restarts": result.metadata.get("restarts", ""),
                "iterations_per_restart": result.metadata.get(
                    "iterations_per_restart", ""
                ),
            })

            # Record objective history for convergence plots.
            for iteration, score in enumerate(result.history):
                history_rows.append({
                    "dataset_id": dataset_id,
                    "trial": trial,
                    "seed": seed,
                    "algorithm": algorithm_name,
                    "iteration": iteration,
                    "current_score": score,
                })

            # Save the initial, final, and best states.
            state_records.append({
                "dataset_id": dataset_id,
                "trial": trial,
                "seed": seed,
                "algorithm": algorithm_name,
                "initial_state": serialize_state(result.initial_state),
                "final_state": serialize_state(result.final_state),
                "best_state": serialize_state(result.best_state),
                "initial_score": initial_score,
                "final_score": final_score,
                "best_score": best_score,
                "runtime_seconds": result.runtime,
                "iterations": result.iterations,
            })

            # Record temperature and acceptance metrics for SA.
            if algorithm_name == "Simulated Annealing":
                metadata = result.metadata

                metric_history = zip(
                    metadata.get("exponential_history", []),
                    metadata.get("temperature_history", []),
                    metadata.get(
                        "acceptance_probability_history", []
                    ),
                )

                for iteration, values in enumerate(
                    metric_history, start=1
                ):
                    (
                        exponential_value,
                        temperature,
                        acceptance_probability,
                    ) = values

                    sa_metric_rows.append({
                        "dataset_id": dataset_id,
                        "trial": trial,
                        "seed": seed,
                        "iteration": iteration,
                        "temperature": temperature,
                        "exp_delta_e_over_t": exponential_value,
                        "acceptance_probability": (
                            acceptance_probability
                        ),
                    })

            print(
                f"Trial {trial} | {algorithm_name} | "
                f"Initial: {initial_score:.2f} | "
                f"Final: {final_score:.2f} | "
                f"Best: {best_score:.2f} | "
                f"Runtime: {result.runtime:.4f}s | "
                f"Iterations: {result.iterations}"
            )

    # Save experiment summaries.
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "dataset_id",
            "trial",
            "seed",
            "algorithm",
            "objective",
            "initial_score",
            "final_score",
            "best_score",
            "fee_score",
            "runtime_seconds",
            "iterations",
            "stuck_count",
            "stuck_episodes",
            "stuck_frequency",
            "stop_reason",
            "sideways_moves",
            "restarts",
            "iterations_per_restart",
        ],
    )

    # Save convergence history.
    write_csv(
        output_dir / "history.csv",
        history_rows,
        [
            "dataset_id",
            "trial",
            "seed",
            "algorithm",
            "iteration",
            "current_score",
        ],
    )

    # Save Simulated Annealing metrics.
    write_csv(
        output_dir / "sa_metrics.csv",
        sa_metric_rows,
        [
            "dataset_id",
            "trial",
            "seed",
            "iteration",
            "temperature",
            "exp_delta_e_over_t",
            "acceptance_probability",
        ],
    )

    # Save states and related experiment details.
    with (output_dir / "states.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            state_records,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSemua hasil eksperimen tersimpan di: {output_dir}")
    print(
        "CATATAN: pastikan dataset yang digunakan sesuai dengan "
        "dataset yang ingin dianalisis."
    )


def main():
    """Parse command-line arguments and run experiments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=(
            "Folder yang berisi ship.csv dan vehicles.csv. "
            "Default mengikuti config.py."
        ),
    )

    parser.add_argument("--bonus", action="store_true",
                        help="Jalankan juga varian HC bonus.")
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="fee",
                        help="fee = objective wajib; eta_weighted = bonus.")
    args = parser.parse_args()
    run_experiments(args.dataset_dir, bonus=args.bonus,
                    objective_name=args.objective)


if __name__ == "__main__":
    main()
