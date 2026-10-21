"""Simulated Annealing for the ship-loading optimization problem.

Efficiency note
---------------
The original version enumerated *all* feasible neighbors at every iteration
(~0.3 s per call on a 10x20 deck with 30 vehicles). SA only needs ONE random
neighbor per iteration, so this version samples a single random move
(swap / move / load-unload / rotate -- the three move families of the spec)
and validates it with the shared ``is_valid_state``. Full neighbor
enumeration is used only (a) as a fallback when sampling finds nothing and
(b) for the periodic local-optimum check that defines "stuck".

Definition of "stuck" (documented for the report)
-------------------------------------------------
Every ``stuck_check_interval`` iterations the *current* state is tested
exactly: if no feasible neighbor has a strictly higher objective, the state is
a local optimum and that check counts as one "stuck" observation.
``stuck_count``    = number of stuck observations,
``stuck_checks``   = number of checks performed,
``stuck_frequency``= stuck_count / stuck_checks,
``stuck_episodes`` = number of maximal runs of consecutive stuck checks.
"""

from math import exp, isfinite
from numbers import Real
from random import Random
from time import perf_counter

from src.core.constraints import is_valid_state
from src.algorithms.moves import propose_random_neighbor
from src.core.models import Ship, State, Vehicle
from src.core.neighbor import get_feasible_neighbors
from src.core.objective import calculate_objective
from src.core.results import AlgorithmResult


def _validate_number(name: str, value: Real) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} harus berupa angka.")
    if not isfinite(value):
        raise ValueError(f"{name} harus berupa angka finite.")


def _is_local_optimum(state, score, vehicles, ship, objective) -> bool:
    return not any(
        objective(n, vehicles) > score
        for n in get_feasible_neighbors(state, vehicles, ship)
    )


def run_simulated_annealing(
    initial_state: State,
    vehicles: dict[str, Vehicle],
    ship: Ship,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.995,
    min_temperature: float = 1e-3,
    max_iterations: int = 5000,
    seed: int | None = 42,
    stuck_check_interval: int = 25,
    objective=calculate_objective,
) -> AlgorithmResult:
    """Run Simulated Annealing (maximization of total ShippingFee).

    Better or equal candidates are always accepted; worse candidates are
    accepted with probability exp(dE / T), dE = f(candidate) - f(current) < 0.
    Geometric cooling: T <- T * cooling_rate.
    """
    _validate_number("initial_temperature", initial_temperature)
    _validate_number("cooling_rate", cooling_rate)
    _validate_number("min_temperature", min_temperature)
    if initial_temperature <= 0:
        raise ValueError("initial_temperature harus lebih besar dari 0.")
    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate harus berada di antara 0 dan 1.")
    if min_temperature <= 0:
        raise ValueError("min_temperature harus lebih besar dari 0.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations harus berupa integer.")
    if max_iterations < 0:
        raise ValueError("max_iterations tidak boleh negatif.")
    if (
        isinstance(stuck_check_interval, bool)
        or not isinstance(stuck_check_interval, int)
        or stuck_check_interval < 1
    ):
        raise ValueError("stuck_check_interval harus integer >= 1.")
    if not is_valid_state(initial_state, vehicles, ship):
        raise ValueError("initial_state tidak valid.")

    start = perf_counter()
    rng = Random(seed)

    current = initial_state.copy()
    current_score = objective(current, vehicles)
    best, best_score = current.copy(), current_score

    history = [current_score]
    temperature_history, exponential_history = [], []
    acceptance_probability_history, accepted_flags = [], []

    stuck_count = stuck_checks = stuck_episodes = 0
    prev_stuck = False
    temperature = initial_temperature
    iterations = 0
    reason = "max_iterations"

    while iterations < max_iterations:
        if temperature <= min_temperature:
            reason = "min_temperature"
            break

        candidate = propose_random_neighbor(current, vehicles, ship, rng)
        if candidate is None:  # sampling failed -> exact check
            all_n = get_feasible_neighbors(current, vehicles, ship)
            if not all_n:
                reason = "no_feasible_neighbors"
                break
            candidate = rng.choice(all_n)

        cand_score = objective(candidate, vehicles)
        delta_e = cand_score - current_score
        exponent = delta_e / temperature
        exp_value = exp(min(exponent, 700))  # e^(dE/T), shown in the plot

        if delta_e >= 0:
            prob, accept = 1.0, True  # improvements recorded as 1.0
        else:
            prob = exp_value
            accept = rng.random() < prob

        temperature_history.append(temperature)
        exponential_history.append(exp_value)
        acceptance_probability_history.append(prob)
        accepted_flags.append(accept)

        if accept:
            current, current_score = candidate, cand_score
            if current_score > best_score:
                best, best_score = current.copy(), current_score

        iterations += 1
        history.append(current_score)
        temperature *= cooling_rate

        if iterations % stuck_check_interval == 0:
            stuck_checks += 1
            is_stuck = _is_local_optimum(
                current, current_score, vehicles, ship, objective
            )
            if is_stuck:
                stuck_count += 1
                if not prev_stuck:
                    stuck_episodes += 1
            prev_stuck = is_stuck

    return AlgorithmResult(
        algorithm="Simulated Annealing",
        initial_state=initial_state.copy(),
        final_state=current.copy(),
        best_state=best.copy(),
        best_score=best_score,
        runtime=perf_counter() - start,
        iterations=iterations,
        history=history,
        metadata={
            "initial_temperature": initial_temperature,
            "cooling_rate": cooling_rate,
            "min_temperature": min_temperature,
            "final_temperature": temperature,
            "max_iterations": max_iterations,
            "seed": seed,
            "termination_reason": reason,
            "stop_reason": reason,
            "temperature_history": temperature_history,
            "exponential_history": exponential_history,
            "acceptance_probability_history": acceptance_probability_history,
            "accepted_history": accepted_flags,
            "acceptance_rate": (
                sum(accepted_flags) / len(accepted_flags) if accepted_flags else 0.0
            ),
            "stuck_check_interval": stuck_check_interval,
            "stuck_checks": stuck_checks,
            "stuck_count": stuck_count,
            "stuck_frequency": (stuck_count / stuck_checks if stuck_checks else 0.0),
            "stuck_episodes": stuck_episodes,
        },
    )
