import csv
import hashlib
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path

from src.algorithms.genetic_algorithm import run_genetic_algorithm
from src.core.constraints import is_valid_state
from src.core.initializer import generate_random_state
from src.core.models import Ship, State, Vehicle
from src.core.objective import calculate_objective


OUTPUT_FILES = (
    "ga_manifest.json",
    "ga_runs.json",
    "ga_summary.csv",
    "ga_history.csv",
)


def serialize_state (state: State) -> dict[str, dict[str, int | str] | None]:
    result: dict[str, dict[str, int | str] | None] = {}
    for vehicle_id in sorted(state.placements):
        placement = state.placements[vehicle_id]
        result[vehicle_id] = (
            None
            if placement is None
            else {
                "x": placement.x,
                "y": placement.y,
                "orientation": placement.orientation.value,
            }
        )
    return result


def calculate_dataset_fingerprint (vehicles: dict[str, Vehicle], ship: Ship) -> str:
    payload = {
        "ship": {
            "width": ship.width,
            "length": ship.length,
            "max_capacity": ship.max_capacity,
        },
        "vehicles": [
            {
                "id": vehicle_id,
                "width": vehicles[vehicle_id].width,
                "length": vehicles[vehicle_id].length,
                "shipping_fee": vehicles[vehicle_id].shipping_fee,
                "weight": vehicles[vehicle_id].weight,
                "eta": vehicles[vehicle_id].eta,
            }
            for vehicle_id in sorted(vehicles)
        ],
    }
    contents = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


def _validate_variations (values: Sequence[int], minimum: int, label: str) -> tuple[int, ...]:
    selected = tuple(values)
    if (
        len(selected) != 3
        or any(type(value) is not int or value < minimum for value in selected)
        or len(set(selected)) != 3
    ):
        raise ValueError(f"{label} must contain three distinct integers >= {minimum}")
    return selected


def _validate_seeds (seeds: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(seeds)
    if len(selected) != 3 or any(type(seed) is not int for seed in selected) or len(set(selected)) != 3:
        raise ValueError("seeds must contain exactly three distinct integers")
    return selected


def _write_csv (path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_ga_parameter_studies (
    vehicles: dict[str, Vehicle],
    ship: Ship,
    *,
    dataset_id: str,
    generation_values: Sequence[int],
    fixed_population_size: int,
    population_values: Sequence[int],
    fixed_generations: int,
    seeds: Sequence[int],
    output_dir: str | Path,
    mutation_rate: float = 0.1,
    load_probability: float = 0.7,
) -> list[dict]:
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a nonempty string")
    generations = _validate_variations(generation_values, 1, "generation_values")
    populations = _validate_variations(population_values, 2, "population_values")
    trials = _validate_seeds(seeds)
    if type(fixed_population_size) is not int or fixed_population_size < 2:
        raise ValueError("fixed_population_size must be an integer >= 2")
    if type(fixed_generations) is not int or fixed_generations < 1:
        raise ValueError("fixed_generations must be a positive integer")
    if (
        isinstance(mutation_rate, bool)
        or not isinstance(mutation_rate, (int, float))
        or not math.isfinite(mutation_rate)
        or not 0 <= mutation_rate <= 1
    ):
        raise ValueError("mutation_rate must be a finite number between 0 and 1")
    if (
        isinstance(load_probability, bool)
        or not isinstance(load_probability, (int, float))
        or not math.isfinite(load_probability)
        or not 0 <= load_probability <= 1
    ):
        raise ValueError("load_probability must be a finite number between 0 and 1")

    empty_state = State({vehicle_id: None for vehicle_id in vehicles})
    if not is_valid_state(empty_state, vehicles, ship):
        raise ValueError("invalid ship or vehicle dataset")

    destination = Path(output_dir)
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    if any((destination / filename).exists() for filename in OUTPUT_FILES):
        raise FileExistsError(f"GA outputs already exist in {destination}")

    fingerprint = calculate_dataset_fingerprint(vehicles, ship)
    initial_states: dict[int, tuple[State, int, int]] = {}
    for seed in trials:
        rng = random.Random(seed)
        initializer_seed = seed
        algorithm_seed = rng.getrandbits(64)
        initial_state = generate_random_state(
            vehicles, ship, load_prob=load_probability, seed=initializer_seed
        )
        if not is_valid_state(initial_state, vehicles, ship):
            raise RuntimeError("random initializer returned an invalid state")
        initial_states[seed] = (initial_state, initializer_seed, algorithm_seed)

    configurations = [
        ("vary_generations", generation, fixed_population_size, generation)
        for generation in generations
    ] + [
        ("vary_population", population, population, fixed_generations)
        for population in populations
    ]

    records: list[dict] = []
    summary_rows: list[dict] = []
    history_rows: list[dict] = []

    for study, setting, population_size, max_generations in configurations:
        for trial, seed in enumerate(trials, start=1):
            initial_state, initializer_seed, algorithm_seed = initial_states[seed]
            result = run_genetic_algorithm (
                initial_state,
                vehicles,
                ship,
                population_size=population_size,
                max_generations=max_generations,
                mutation_rate=mutation_rate,
                seed=algorithm_seed,
            )
            mean_history = result.metadata["average_fitness_history"]
            if (
                len(result.history) != max_generations + 1
                or len(mean_history) != len(result.history)
                or result.best_score != max(result.history)
                or any(not is_valid_state(state, vehicles, ship) for state in (
                    result.initial_state, result.final_state, result.best_state
                ))
            ):
                raise RuntimeError("GA output violated experiment invariants")

            run_id = f"{study}_{setting}_trial_{trial}"
            final_score = calculate_objective(result.final_state, vehicles)
            record = {
                "run_id": run_id,
                "study": study,
                "setting": setting,
                "trial": trial,
                "dataset_id": dataset_id,
                "dataset_sha256": fingerprint,
                "seed": seed,
                "initializer_seed": initializer_seed,
                "algorithm_seed": algorithm_seed,
                "population_size": population_size,
                "max_generations": max_generations,
                "mutation_rate": mutation_rate,
                "load_probability": load_probability,
                "initial_score": calculate_objective(initial_state, vehicles),
                "final_score": final_score,
                "best_score": result.best_score,
                "best_generation": result.metadata["best_generation"],
                "fitness_evaluations": result.metadata["fitness_evaluations"],
                "runtime_seconds": result.runtime,
                "initial_state": serialize_state(result.initial_state),
                "final_state": serialize_state(result.final_state),
                "best_state": serialize_state(result.best_state),
                "maximum_fitness_history": list(result.history),
                "average_fitness_history": list(mean_history),
            }
            records.append(record)
            summary_rows.append({
                field: value
                for field, value in record.items()
                if field not in (
                    "initial_state", "final_state", "best_state",
                    "maximum_fitness_history", "average_fitness_history",
                )
            })
            for generation, (maximum, average) in enumerate(zip(result.history, mean_history)):
                history_rows.append({
                    "run_id": run_id,
                    "study": study,
                    "setting": setting,
                    "trial": trial,
                    "seed": seed,
                    "population_size": population_size,
                    "max_generations": max_generations,
                    "generation": generation,
                    "maximum_fitness": maximum,
                    "average_fitness": average,
                })

    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_sha256": fingerprint,
        "study_generation_values": list(generations),
        "study_fixed_population_size": fixed_population_size,
        "study_population_values": list(populations),
        "study_fixed_generations": fixed_generations,
        "mutation_rate": mutation_rate,
        "load_probability": load_probability,
        "seeds": list(trials),
        "number_of_runs": len(records),
    }
    destination.mkdir(parents=True, exist_ok=True)
    for filename, payload in (("ga_manifest.json", manifest), ("ga_runs.json", records)):
        with (destination / filename).open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")

    summary_fields = (
        "run_id", "study", "setting", "trial", "dataset_id", "dataset_sha256",
        "seed", "initializer_seed", "algorithm_seed", "population_size",
        "max_generations", "mutation_rate", "load_probability", "initial_score", "final_score",
        "best_score", "best_generation", "fitness_evaluations", "runtime_seconds",
    )
    history_fields = (
        "run_id", "study", "setting", "trial", "seed", "population_size",
        "max_generations", "generation", "maximum_fitness", "average_fitness",
    )
    _write_csv(destination / "ga_summary.csv", summary_fields, summary_rows)
    _write_csv(destination / "ga_history.csv", history_fields, history_rows)
    return records