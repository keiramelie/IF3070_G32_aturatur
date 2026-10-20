
"""Tests for Hill Climbing and Simulated Annealing."""

import pytest

from src.algorithms.hill_climbing import run_hill_climbing
from src.algorithms.simulated_annealing import run_simulated_annealing
from src.core.constraints import is_valid_state
from src.core.models import Ship, State, Vehicle
from src.core.objective import calculate_objective


@pytest.fixture
def problem():
    """Create a small, predictable ship-loading problem."""
    ship = Ship(width=3, length=3, max_capacity=20)

    vehicles = {
        "A": Vehicle("A", 1, 1, 100, 5, 1),
        "B": Vehicle("B", 1, 1, 80, 5, 2),
        "C": Vehicle("C", 1, 1, 60, 5, 3),
    }

    initial_state = State(
        placements={
            vehicle_id: None
            for vehicle_id in vehicles
        }
    )

    return ship, vehicles, initial_state


def test_hill_climbing_returns_valid_result(problem):
    ship, vehicles, initial_state = problem

    result = run_hill_climbing(
        initial_state,
        vehicles,
        ship,
        max_iterations=50,
    )

    assert is_valid_state(result.final_state, vehicles, ship)
    assert is_valid_state(result.best_state, vehicles, ship)

    assert result.best_score == calculate_objective(
        result.best_state, vehicles
    )
    assert result.best_score >= calculate_objective(
        initial_state, vehicles
    )

    # The algorithm must not modify the caller's initial state.
    assert all(
        placement is None
        for placement in initial_state.placements.values()
    )

    assert result.iterations <= 50
    assert len(result.history) <= result.iterations + 1
    assert result.history[0] == calculate_objective(
        result.initial_state, vehicles
    )
    assert result.history[-1] == calculate_objective(
        result.final_state, vehicles
    )


def test_hill_climbing_finds_best_loading_for_small_problem(problem):
    ship, vehicles, initial_state = problem

    result = run_hill_climbing(
        initial_state,
        vehicles,
        ship,
        max_iterations=50,
    )

    expected_score = sum(
        vehicle.shipping_fee
        for vehicle in vehicles.values()
    )

    # All three vehicles fit, and the ship has enough capacity.
    assert result.best_score == expected_score
    assert len(result.best_state.loaded_vehicle_ids()) == 3


def test_hill_climbing_zero_iterations(problem):
    ship, vehicles, initial_state = problem

    result = run_hill_climbing(
        initial_state,
        vehicles,
        ship,
        max_iterations=0,
    )

    assert result.iterations == 0
    assert result.best_score == 0
    assert result.history == [0.0]


def test_simulated_annealing_returns_valid_result(problem):
    ship, vehicles, initial_state = problem

    result = run_simulated_annealing(
        initial_state,
        vehicles,
        ship,
        max_iterations=100,
        seed=42,
    )

    assert is_valid_state(result.final_state, vehicles, ship)
    assert is_valid_state(result.best_state, vehicles, ship)

    assert result.best_score == calculate_objective(
        result.best_state, vehicles
    )
    assert result.best_score >= calculate_objective(
        initial_state, vehicles
    )

    assert result.iterations <= 100
    assert len(result.history) == result.iterations + 1

    # The algorithm must not modify the caller's initial state.
    assert all(
        placement is None
        for placement in initial_state.placements.values()
    )

    # New checks for Simulated Annealing metrics.
    metadata = result.metadata

    assert len(metadata["exponential_history"]) == result.iterations
    assert len(metadata["temperature_history"]) == result.iterations
    assert (
        len(metadata["acceptance_probability_history"])
        == result.iterations
    )

    assert all(
        0.0 <= probability <= 1.0
        for probability in metadata["acceptance_probability_history"]
    )


def test_simulated_annealing_is_reproducible(problem):
    ship, vehicles, initial_state = problem

    result_one = run_simulated_annealing(
        initial_state,
        vehicles,
        ship,
        max_iterations=100,
        seed=42,
    )

    result_two = run_simulated_annealing(
        initial_state,
        vehicles,
        ship,
        max_iterations=100,
        seed=42,
    )

    assert result_one.final_state == result_two.final_state
    assert result_one.best_state == result_two.best_state
    assert result_one.history == result_two.history
    assert result_one.best_score == result_two.best_score

    assert (
        result_one.metadata["acceptance_probability_history"]
        == result_two.metadata["acceptance_probability_history"]
    )


def test_simulated_annealing_zero_iterations(problem):
    ship, vehicles, initial_state = problem

    result = run_simulated_annealing(
        initial_state,
        vehicles,
        ship,
        max_iterations=0,
        seed=42,
    )

    assert result.iterations == 0
    assert result.best_score == 0
    assert result.history == [0.0]

    assert result.metadata["exponential_history"] == []
    assert result.metadata["temperature_history"] == []
    assert result.metadata["acceptance_probability_history"] == []


def test_simulated_annealing_rejects_invalid_temperature(problem):
    ship, vehicles, initial_state = problem

    with pytest.raises(ValueError):
        run_simulated_annealing(
            initial_state,
            vehicles,
            ship,
            initial_temperature=0,
        )


def test_simulated_annealing_rejects_invalid_cooling_rate(problem):
    ship, vehicles, initial_state = problem

    with pytest.raises(ValueError):
        run_simulated_annealing(
            initial_state,
            vehicles,
            ship,
            cooling_rate=1.0,
        )


# --- Bonus HC variants & shared random-move sampler -------------------------
from random import Random

from src.algorithms.hill_climbing import (
    run_hill_climbing_random_restart,
    run_hill_climbing_sideways,
    run_hill_climbing_stochastic,
)
from src.algorithms.moves import propose_random_neighbor


def _random_problem():
    ship = Ship(width=5, length=5, max_capacity=30)
    vehicles = {
        "A": Vehicle("A", 2, 2, 100, 6, 1),
        "B": Vehicle("B", 1, 2, 70, 4, 2),
        "C": Vehicle("C", 3, 1, 60, 5, 3),
        "D": Vehicle("D", 1, 1, 30, 2, 4),
    }
    return ship, vehicles, State({k: None for k in vehicles})


def test_propose_random_neighbor_is_valid_and_pure():
    ship, vehicles, state = _random_problem()
    rng = Random(0)
    before = dict(state.placements)
    for _ in range(200):
        cand = propose_random_neighbor(state, vehicles, ship, rng)
        assert cand is not None
        assert is_valid_state(cand, vehicles, ship)
        assert cand.placements != before
        state = cand
    # input state of the first call was never mutated
    assert all(v is None for v in before.values())


@pytest.mark.parametrize("runner", [
    lambda s, v, sh: run_hill_climbing_sideways(s, v, sh, max_sideways=3, seed=1),
    lambda s, v, sh: run_hill_climbing_stochastic(s, v, sh, seed=1),
    lambda s, v, sh: run_hill_climbing_random_restart(
        s, v, sh, max_restarts=2, seed=1),
])
def test_hc_variants_contract(runner):
    ship, vehicles, initial = _random_problem()
    result = runner(initial, vehicles, ship)
    assert is_valid_state(result.final_state, vehicles, ship)
    assert is_valid_state(result.best_state, vehicles, ship)
    assert result.best_score == calculate_objective(result.best_state, vehicles)
    assert result.best_score >= 100
    assert all(p is None for p in initial.placements.values())
    assert "stop_reason" in result.metadata


def test_sideways_respects_limit():
    ship, vehicles, initial = _random_problem()
    r = run_hill_climbing_sideways(initial, vehicles, ship, max_sideways=2, seed=3)
    assert r.metadata["sideways_moves"] <= 2


def test_sa_stuck_metadata_and_reproducible():
    ship, vehicles, initial = _random_problem()
    kw = dict(initial_temperature=50, cooling_rate=0.9, max_iterations=100,
              seed=7, stuck_check_interval=10)
    a = run_simulated_annealing(initial, vehicles, ship, **kw)
    b = run_simulated_annealing(initial, vehicles, ship, **kw)
    assert a.history == b.history
    assert a.metadata["stuck_checks"] == a.iterations // 10
    assert 0 <= a.metadata["stuck_frequency"] <= 1


# --- Bonus: alternative objective ------------------------------------------
from src.algorithms.objectives import OBJECTIVES, eta_weighted_objective


def test_eta_weighted_equals_fee_when_weight_zero():
    ship, vehicles, _ = _random_problem()
    st = State({"A": None, "B": None, "C": None, "D": None})
    r = run_hill_climbing(st, vehicles, ship, max_iterations=20)
    assert eta_weighted_objective(r.best_state, vehicles, 0.0) == calculate_objective(
        r.best_state, vehicles
    )


def test_eta_weighted_prefers_urgent_vehicle():
    # Sama fee, tetapi A mendesak (ETA 1) dan B tidak (ETA 10).
    vehicles = {
        "A": Vehicle("A", 1, 1, 100, 1, 1),
        "B": Vehicle("B", 1, 1, 100, 1, 10),
    }
    only_a = State({"A": None, "B": None})
    from src.core.models import Orientation, Placement
    only_a.placements["A"] = Placement(0, 0, Orientation.HORIZONTAL)
    only_b = State({"A": None, "B": Placement(0, 0, Orientation.HORIZONTAL)})
    assert eta_weighted_objective(only_a, vehicles) > eta_weighted_objective(only_b, vehicles)
    assert calculate_objective(only_a, vehicles) == calculate_objective(only_b, vehicles)


def test_algorithms_accept_custom_objective():
    ship, vehicles, initial = _random_problem()
    obj = OBJECTIVES["eta_weighted"]
    for res in (
        run_hill_climbing(initial, vehicles, ship, 20, objective=obj),
        run_simulated_annealing(initial, vehicles, ship, max_iterations=50,
                                seed=1, objective=obj),
    ):
        assert is_valid_state(res.best_state, vehicles, ship)
        assert res.best_score == obj(res.best_state, vehicles)
