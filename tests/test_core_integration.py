
import random

from src.core.models import Vehicle, Ship
from src.core.initializer import generate_random_state
from src.core.objective import calculate_objective
from src.core.constraints import is_valid_state
from src.core.grid import build_grid
from src.core.neighbor import (
    get_feasible_neighbors,
    get_random_feasible_neighbor,
)


def test_core_integration():
    # 1. Prepare the problem dataset
    ship = Ship(width=8, length=8, max_capacity=20)

    vehicles = {
        "A": Vehicle("A", 2, 3, 60, 5, 2),
        "B": Vehicle("B", 3, 4, 90, 8, 1),
        "C": Vehicle("C", 2, 2, 40, 4, 3),
        "D": Vehicle("D", 3, 3, 70, 6, 4),
    }

    # 2. Generate an initial state
    initial_state = generate_random_state(
        vehicles,
        ship,
        load_prob=0.7,
        seed=42,
    )

    assert is_valid_state(initial_state, vehicles, ship)
    assert set(initial_state.placements) == set(vehicles)

    # 3. Calculate the initial objective
    initial_score = calculate_objective(initial_state, vehicles)

    expected_score = sum(
        vehicles[vehicle_id].shipping_fee
        for vehicle_id in initial_state.loaded_vehicle_ids()
    )

    assert initial_score == expected_score

    # 4. Build the ship grid
    grid = build_grid(initial_state, vehicles, ship)

    assert len(grid) == ship.length
    assert all(len(row) == ship.width for row in grid)

    # 5. Generate feasible neighbors
    neighbors = get_feasible_neighbors(
        initial_state, vehicles, ship
    )

    assert len(neighbors) > 0

    for neighbor in neighbors:
        assert is_valid_state(neighbor, vehicles, ship)
        assert neighbor != initial_state

        score = calculate_objective(neighbor, vehicles)
        assert score >= 0

        neighbor_grid = build_grid(neighbor, vehicles, ship)
        assert len(neighbor_grid) == ship.length

    # 6. Random neighbor must belong to feasible neighbors
    random_neighbor = get_random_feasible_neighbor(
        initial_state,
        vehicles,
        ship,
        rng=random.Random(42),
    )

    assert random_neighbor in neighbors

    # 7. Initial state must remain unchanged
    assert calculate_objective(initial_state, vehicles) == initial_score
    assert is_valid_state(initial_state, vehicles, ship)

    print("All core integration tests passed")
    print(f"Initial objective: {initial_score}")
    print(f"Feasible neighbors: {len(neighbors)}")


if __name__ == "__main__":
    test_core_integration()
