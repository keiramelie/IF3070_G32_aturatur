
"""Steepest-Ascent Hill Climbing for the ship-loading optimizer."""

from time import perf_counter

from src.core.models import Vehicle, Ship, State
from src.core.objective import calculate_objective
from src.core.constraints import is_valid_state
from src.core.neighbor import get_feasible_neighbors
from src.core.results import AlgorithmResult


def run_hill_climbing(
    initial_state: State,
    vehicles: dict[str, Vehicle],
    ship: Ship,
    max_iterations: int | None = None,
    objective=calculate_objective,
) -> AlgorithmResult:
    """Run Steepest-Ascent Hill Climbing.

    At each iteration, evaluate all feasible neighbors and select the
    highest-scoring one. Move to it only if it improves the objective.

    Args:
        initial_state: Valid starting state.
        vehicles: Mapping of vehicle IDs to Vehicle objects.
        ship: Ship and deck constraints.
        max_iterations: Maximum number of neighbor evaluations, or None
            to continue until a stopping condition is reached.

    Returns:
        AlgorithmResult containing the final state, best state, score
        history, runtime, iteration count, and termination metadata.
    """
    if max_iterations is not None:
        if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
        ):
            raise TypeError("max_iterations must be an integer or None")

        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")

    if not is_valid_state(initial_state, vehicles, ship):
        raise ValueError("initial_state must be valid")

    start_time = perf_counter()

    # Keep the input state unchanged.
    initial_copy = initial_state.copy()
    current_state = initial_state.copy()
    current_score = objective(current_state, vehicles)

    best_state = current_state.copy()
    best_score = current_score

    # history[0] represents the initial objective.
    history = [current_score]

    iterations = 0
    termination_reason = "max_iterations"

    while max_iterations is None or iterations < max_iterations:
        neighbors = get_feasible_neighbors(
            current_state,
            vehicles,
            ship,
        )

        if not neighbors:
            termination_reason = "no_feasible_neighbors"
            break

        # Steepest-Ascent: evaluate all feasible neighbors and select
        # the neighbor with the highest objective value.
        neighbor_scores = [
            objective(neighbor, vehicles)
            for neighbor in neighbors
        ]

        best_index = max(
            range(len(neighbors)),
            key=lambda index: neighbor_scores[index],
        )

        best_neighbor = neighbors[best_index]
        neighbor_score = neighbor_scores[best_index]

        iterations += 1

        # Stop when no neighbor improves the current objective.
        if neighbor_score <= current_score:
            # Record the completed iteration while preserving the
            # current state. This keeps history aligned with iterations.
            history.append(current_score)
            termination_reason = "local_optimum"
            break

        # Move to the best improving neighbor.
        current_state = best_neighbor.copy()
        current_score = neighbor_score

        if current_score > best_score:
            best_state = current_state.copy()
            best_score = current_score

        history.append(current_score)

    runtime = perf_counter() - start_time

    return AlgorithmResult(
        algorithm="Hill Climbing",
        initial_state=initial_copy,
        final_state=current_state.copy(),
        best_state=best_state.copy(),
        best_score=best_score,
        runtime=runtime,
        iterations=iterations,
        history=history,
        metadata={
            "variant": "Steepest-Ascent",
            "max_iterations": max_iterations,
            "termination_reason": termination_reason,
            "stop_reason": termination_reason,
        },
    )


# ---------------------------------------------------------------------------
# Bonus: other Hill Climbing variants (sideways, stochastic, random restart)
# ---------------------------------------------------------------------------
from random import Random  # noqa: E402

from src.core.initializer import generate_random_state  # noqa: E402


def _finish(name, initial, current, best, best_score, start, iterations,
            history, metadata):
    return AlgorithmResult(
        algorithm=name,
        initial_state=initial.copy(),
        final_state=current.copy(),
        best_state=best.copy(),
        best_score=best_score,
        runtime=perf_counter() - start,
        iterations=iterations,
        history=history,
        metadata=metadata,
    )


def run_hill_climbing_sideways(
    initial_state, vehicles, ship, max_sideways=20, max_iterations=1000,
    seed=None, objective=calculate_objective,
) -> AlgorithmResult:
    """Steepest-ascent that may take equal-score (sideways) moves.

    Stops at a strict local optimum, when ``max_sideways`` sideways moves
    have been used, or at ``max_iterations``.
    """
    if not is_valid_state(initial_state, vehicles, ship):
        raise ValueError("initial_state must be valid")
    if max_sideways < 0:
        raise ValueError("max_sideways must be non-negative")
    start, rng = perf_counter(), Random(seed)
    current = initial_state.copy()
    score = objective(current, vehicles)
    best, best_score = current.copy(), score
    history, iterations, sideways = [score], 0, 0
    reason = "max_iterations"
    while iterations < max_iterations:
        neighbors = get_feasible_neighbors(current, vehicles, ship)
        if not neighbors:
            reason = "no_feasible_neighbors"
            break
        scores = [objective(n, vehicles) for n in neighbors]
        top = max(scores)
        iterations += 1
        if top < score:
            history.append(score)
            reason = "local_optimum"
            break
        if top == score:
            if sideways >= max_sideways:
                history.append(score)
                reason = "max_sideways"
                break
            sideways += 1
        current = rng.choice(
            [n for n, s in zip(neighbors, scores) if s == top]
        )
        score = top
        if score > best_score:
            best, best_score = current.copy(), score
        history.append(score)
    return _finish(
        "Hill Climbing (Sideways)", initial_state, current, best, best_score,
        start, iterations, history,
        {"variant": "Sideways Move", "max_sideways": max_sideways,
         "sideways_moves": sideways, "max_iterations": max_iterations,
         "seed": seed, "termination_reason": reason, "stop_reason": reason},
    )


def run_hill_climbing_stochastic(
    initial_state, vehicles, ship, max_iterations=1000, seed=None,
    objective=calculate_objective,
) -> AlgorithmResult:
    """Stochastic HC: move to a RANDOM strictly-improving neighbor."""
    if not is_valid_state(initial_state, vehicles, ship):
        raise ValueError("initial_state must be valid")
    start, rng = perf_counter(), Random(seed)
    current = initial_state.copy()
    score = objective(current, vehicles)
    best, best_score = current.copy(), score
    history, iterations = [score], 0
    reason = "max_iterations"
    while iterations < max_iterations:
        neighbors = get_feasible_neighbors(current, vehicles, ship)
        improving = [
            (n, s) for n in neighbors
            if (s := objective(n, vehicles)) > score
        ]
        iterations += 1
        if not improving:
            history.append(score)
            reason = "local_optimum" if neighbors else "no_feasible_neighbors"
            break
        current, score = rng.choice(improving)
        if score > best_score:
            best, best_score = current.copy(), score
        history.append(score)
    return _finish(
        "Hill Climbing (Stochastic)", initial_state, current, best,
        best_score, start, iterations, history,
        {"variant": "Stochastic", "max_iterations": max_iterations,
         "seed": seed, "termination_reason": reason, "stop_reason": reason},
    )


def run_hill_climbing_random_restart(
    initial_state, vehicles, ship, max_restarts=5, max_iterations=1000,
    load_prob=0.5, seed=None, objective=calculate_objective,
) -> AlgorithmResult:
    """Random-restart HC: steepest-ascent from several random states.

    Restart 0 uses ``initial_state``; later restarts use fresh random states.
    Stops after ``max_restarts`` restarts (in addition to the first run).
    """
    if not is_valid_state(initial_state, vehicles, ship):
        raise ValueError("initial_state must be valid")
    if max_restarts < 0:
        raise ValueError("max_restarts must be non-negative")
    start = perf_counter()
    seed_rng = Random(seed)
    best, best_score, best_final = None, float("-inf"), None
    history, per_restart, iterations = [], [], 0
    scores_per_restart = []
    for r in range(max_restarts + 1):
        s0 = (initial_state if r == 0 else generate_random_state(
            vehicles, ship, load_prob=load_prob,
            seed=seed_rng.randrange(2**31)))
        res = run_hill_climbing(s0, vehicles, ship, max_iterations, objective)
        history.extend(res.history)
        per_restart.append(res.iterations)
        scores_per_restart.append(res.best_score)
        iterations += res.iterations
        if res.best_score > best_score:
            best, best_score, best_final = (
                res.best_state, res.best_score, res.final_state)
    return _finish(
        "Hill Climbing (Random Restart)", initial_state, best_final, best,
        best_score, start, iterations, history,
        {"variant": "Random Restart", "restarts": max_restarts,
         "max_restarts": max_restarts,
         "iterations_per_restart": per_restart,
         "best_score_per_restart": scores_per_restart,
         "max_iterations": max_iterations, "seed": seed,
         "termination_reason": "max_restarts",
         "stop_reason": "max_restarts"},
    )
