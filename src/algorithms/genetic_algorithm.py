"""Genetic Algorithm"""

import math
import random
import time
from collections.abc import Sequence

from src.core.constraints import is_valid_state, is_within_bounds
from src.core.initializer import generate_random_state
from src.core.models import Orientation, Placement, Ship, State, Vehicle
from src.core.objective import calculate_objective
from src.core.results import AlgorithmResult


MOVE_UNLOAD_PROBABILITY = 0.2

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


def select_parent_index (
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


def crossover_one_point (
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


def repair_state (
    candidate: State,
    vehicles: dict[str, Vehicle],
    ship: Ship,
    rng: random.Random,
) -> State:
    if set(candidate.placements) != set(vehicles):
        raise ValueError("candidate must contain exactly the vehicle IDs")

    for placement in candidate.placements.values():
        if placement is None:
            continue
        if (
            not isinstance(placement, Placement)
            or type(placement.x) is not int
            or type(placement.y) is not int
            or not isinstance(placement.orientation, Orientation)
        ):
            raise ValueError("candidate contains a malformed placement")

    repaired = State({vehicle_id: None for vehicle_id in vehicles})
    if not is_valid_state(repaired, vehicles, ship):
        raise ValueError("ship or vehicle data cannot form a feasible empty state")

    occupied: set[tuple[int, int]] = set()
    used_capacity = 0.0

    def free_cells(vehicle: Vehicle, placement: Placement) -> bool:
        if not is_within_bounds(placement, vehicle, ship):
            return False
        width, length = placement.dimension(vehicle)
        return all(
            (x, y) not in occupied
            for y in range(placement.y, placement.y + length)
            for x in range(placement.x, placement.x + width)
        )

    def place(vehicle_id: str, placement: Placement) -> None:
        nonlocal used_capacity
        repaired.placements[vehicle_id] = placement
        used_capacity += vehicles[vehicle_id].weight
        width, length = placement.dimension(vehicles[vehicle_id])
        occupied.update(
            (x, y)
            for y in range(placement.y, placement.y + length)
            for x in range(placement.x, placement.x + width)
        )

    loaded_ids = sorted(
        (vehicle_id for vehicle_id in vehicles if candidate.placements[vehicle_id] is not None),
        key=lambda vehicle_id: (-vehicles[vehicle_id].shipping_fee, vehicle_id),
    )
    for vehicle_id in loaded_ids:
        vehicle = vehicles[vehicle_id]
        inherited = candidate.placements[vehicle_id]
        assert inherited is not None

        if used_capacity + vehicle.weight > ship.max_capacity:
            continue

        if free_cells(vehicle, inherited):
            place(vehicle_id, inherited)
            continue

        other_orientation = (
            Orientation.VERTICAL
            if inherited.orientation == Orientation.HORIZONTAL
            else Orientation.HORIZONTAL
        )
        relocated = False
        for orientation in (inherited.orientation, other_orientation):
            probe = Placement(0, 0, orientation)
            width, length = probe.dimension(vehicle)
            columns = ship.width - width + 1
            rows = ship.length - length + 1
            if columns <= 0 or rows <= 0:
                continue

            count = columns * rows
            start = rng.randrange(count)
            for offset in range(count):
                index = (start + offset) % count
                proposed = Placement(index % columns, index // columns, orientation)
                if proposed != inherited and free_cells(vehicle, proposed):
                    place(vehicle_id, proposed)
                    relocated = True
                    break
            if relocated:
                break

    if not is_valid_state(repaired, vehicles, ship):
        raise RuntimeError("repair generated an invalid state; core invariants violated")
    return repaired

def mutate_state (
    state: State,
    vehicles: dict[str, Vehicle],
    ship: Ship,
    rng: random.Random,
    *,
    max_attempts: int = 32,
) -> State:
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if not is_valid_state(state, vehicles, ship):
        raise ValueError("state must be feasible before mutation")

    vehicle_ids = tuple(sorted(vehicles))
    if not vehicle_ids:
        return state.copy()

    loaded_ids = tuple(
        vehicle_id for vehicle_id in vehicle_ids
        if state.placements[vehicle_id] is not None
    )
    available_operations = ["move"]
    if len(vehicle_ids) > 1:
        available_operations.append("swap")
    if loaded_ids:
        available_operations.append("rotate")

    for _ in range(max_attempts):
        operation = rng.choice(available_operations)
        proposal = state.copy()
        if operation == "swap":
            a, b = rng.sample(vehicle_ids, 2)
            proposal.placements[a], proposal.placements[b] = (
                state.placements[b], state.placements[a]
            )
        elif operation == "rotate":
            vehicle_id = rng.choice(loaded_ids)
            current = state.placements[vehicle_id]
            assert current is not None
            new_orientation = (
                Orientation.VERTICAL
                if current.orientation == Orientation.HORIZONTAL
                else Orientation.HORIZONTAL
            )
            proposal.placements[vehicle_id] = Placement(
                current.x, current.y, new_orientation
            )
        else:
            vehicle_id = rng.choice(vehicle_ids)
            current = state.placements[vehicle_id]
            if current is not None and rng.random() < MOVE_UNLOAD_PROBABILITY:
                proposal.placements[vehicle_id] = None
            else:
                orientation = (
                    rng.choice(tuple(Orientation))
                    if current is None else current.orientation
                )
                width, length = Placement(0, 0, orientation).dimension(vehicles[vehicle_id])
                columns = ship.width - width + 1
                rows = ship.length - length + 1
                if columns <= 0 or rows <= 0:
                    continue
                proposal.placements[vehicle_id] = Placement(
                    x=rng.randrange(columns),
                    y=rng.randrange(rows),
                    orientation=orientation,
                )
    
        if proposal.placements != state.placements and is_valid_state(
            proposal, vehicles, ship
        ):
            return proposal
        
    return state.copy()

def run_genetic_algorithm (
    initial_state: State,
    vehicles: dict[str, Vehicle],
    ship: Ship,
    population_size: int = 50,
    max_generations: int = 200,
    mutation_rate: float = 0.1,
    seed: int | None = None,
) -> AlgorithmResult:
    if type(max_generations) is not int or max_generations < 0:
        raise ValueError("max_generations must be a nonnegative integer")
    if (
        isinstance(mutation_rate, bool)
        or not isinstance(mutation_rate, (int, float))
        or not math.isfinite(mutation_rate)
        or not 0.0 <= mutation_rate <= 1.0
    ):
        raise ValueError("mutation_rate must be a finite number between 0 and 1")

    rng = random.Random(seed)
    start = time.perf_counter()
    population = initialize_population(
        initial_state,
        vehicles,
        ship,
        population_size,
        seed=rng.getrandbits(64),
    )
    vehicle_ids = tuple(sorted(vehicles))
    fitnesses, maximum, average = evaluate_population(population, vehicles, ship)
    history = [maximum]
    average_fitness_history = [average]
    best_index = max(range(population_size), key=fitnesses.__getitem__)
    best_state = population[best_index].copy()
    best_score = fitnesses[best_index]
    best_generation = 0

    for generation in range(1, max_generations + 1):
        next_population: list[State] = []
        for _ in range(population_size):
            parent_a = population[select_parent_index(fitnesses, rng)]
            parent_b = population[select_parent_index(fitnesses, rng)]
            child = crossover_one_point(parent_a, parent_b, vehicle_ids, rng)
            child = repair_state(child, vehicles, ship, rng)
            if rng.random() < mutation_rate:
                child = mutate_state(child, vehicles, ship, rng)
            next_population.append(child)

        population = next_population
        fitnesses, maximum, average = evaluate_population(population, vehicles, ship)
        history.append(maximum)
        average_fitness_history.append(average)
        generation_best_index = max(range(population_size), key=fitnesses.__getitem__)
        generation_best_score = fitnesses[generation_best_index]
        if generation_best_score > best_score:
            best_state = population[generation_best_index].copy()
            best_score = generation_best_score
            best_generation = generation

    final_index = max(range(population_size), key=fitnesses.__getitem__)
    final_state = population[final_index].copy()
    return AlgorithmResult(
        algorithm="Genetic Algorithm",
        initial_state=initial_state.copy(),
        final_state=final_state,
        best_state=best_state,
        best_score=best_score,
        runtime=time.perf_counter() - start,
        iterations=max_generations,
        history=history,
        metadata={
            "population_size": population_size,
            "max_generations": max_generations,
            "mutation_rate": mutation_rate,
            "seed": seed,
            "average_fitness_history": average_fitness_history,
            "best_generation": best_generation,
            "final_population_best_score": fitnesses[final_index],
            "fitness_evaluations": population_size * (max_generations + 1),
        },
    )