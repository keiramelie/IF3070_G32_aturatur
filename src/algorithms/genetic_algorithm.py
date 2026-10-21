"""Genetic Algorithm: population setup and fitness tracking."""

import random
from collections.abc import Sequence

from src.core.constraints import is_valid_state
from src.core.initializer import generate_random_state
from src.core.models import Ship, State, Vehicle
from src.core.objective import calculate_objective

def initialize_population(
    initial_state: State,
    vehicles: dict[str, Vehicle],
    ship: Ship,
    population_size: int,
    *,
    seed: int | None = None,
) -> list[State]:
    """Keep a copy of the supplied state and create other feasible individuals."""
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


def evaluate_population(
    population: Sequence[State],
    vehicles: dict[str, Vehicle],
    ship: Ship,
) -> tuple[list[float], float, float]:
    """Return individual fitnesses, maximum fitness, and mean fitness."""
    if not population:
        raise ValueError("population cannot be empty")

    fitnesses: list[float] = []
    for individual in population:
        if not is_valid_state(individual, vehicles, ship):
            raise ValueError("population contains an infeasible state")
        fitnesses.append(calculate_objective(individual, vehicles))

    return fitnesses, max(fitnesses), sum(fitnesses) / len(fitnesses)