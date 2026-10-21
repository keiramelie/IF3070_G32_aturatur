"""Genetic Algorithm"""

import math
import random
from collections.abc import Sequence

from src.core.constraints import is_valid_state
from src.core.initializer import generate_random_state
from src.core.models import Ship, State, Vehicle
from src.core.objective import calculate_objective


def initialize_population (
    initial_state: State,
    vehicles: dict[str, Vehicle],
    ship: Ship,
    population_size: int,
    *,
    seed: int | None = None,
) -> list[State]:
    if type(population_size) is not int or population_size < 2:
        raise ValueError("population_size must be an integer >= 2")
    if not is_valid_state(initial_state, vehicles, ship):
        raise ValueError("initial_state must be feasible")

    rng = random.Random(seed)
    population = [initial_state.copy()]
    for _ in range(population_size - 1):
        individual = generate_random_state(
            vehicles, ship, seed=rng.getrandbits(64)
        )
        if not is_valid_state(individual, vehicles, ship):
            raise RuntimeError("core initializer produced an infeasible state")
        population.append(individual)
    return population


def evaluate_population (
    population: Sequence[State],
    vehicles: dict[str, Vehicle],
    ship: Ship,
) -> tuple[list[float], float, float]:
    if not population:
        raise ValueError("population cannot be empty")

    fitnesses: list[float] = []
    for individual in population:
        if not is_valid_state(individual, vehicles, ship):
            raise ValueError("population contains an infeasible state")
        fitnesses.append(calculate_objective(individual, vehicles))

    return fitnesses, max(fitnesses), sum(fitnesses) / len(fitnesses)


def select_parent_index(
    fitnesses: Sequence[float],
    rng: random.Random,
) -> int:
    if not fitnesses:
        raise ValueError("fitnesses cannot be empty")
    if any(
        isinstance(fitness, bool)
        or not isinstance(fitness, (int, float))
        or not math.isfinite(fitness)
        or fitness < 0
        for fitness in fitnesses
    ):
        raise ValueError("fitnesses must be finite, nonnegative numbers")

    if not any(fitnesses):
        return rng.randrange(len(fitnesses))

    return rng.choices(range(len(fitnesses)), weights=fitnesses, k=1)[0]


def crossover_one_point(
    parent_a: State,
    parent_b: State,
    vehicle_ids: Sequence[str],
    rng: random.Random,
) -> State:
    ids = tuple(vehicle_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("vehicle_ids must not contain duplicates")
    required = set(ids)
    if set(parent_a.placements) != required or set(parent_b.placements) != required:
        raise ValueError("both parents must contain exactly the vehicle IDs")

    if len(ids) < 2:
        return parent_a.copy()

    cut = rng.randrange(1, len(ids))
    return State(
        placements={
            vehicle_id: (
                parent_a.placements[vehicle_id]
                if index < cut
                else parent_b.placements[vehicle_id]
            )
            for index, vehicle_id in enumerate(ids)
        }
    )
