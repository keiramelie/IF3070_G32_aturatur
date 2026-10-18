
import random

from src.core.models import (
    Vehicle, Ship, Placement, State, Orientation
)
from src.core.constraints import is_valid_state
from src.core.neighbor import (
    get_feasible_neighbors,
    get_random_feasible_neighbor
)


def test_neighbor():
    ship = Ship(
        width=5,
        length=4,
        max_capacity=20
    )

    vehicles = {
        "A": Vehicle("A", 2, 3, 60, 5, 2),
        "B": Vehicle("B", 2, 3, 40, 6, 3),
        "C": Vehicle("C", 2, 2, 30, 4, 1),
    }

    original = State({
        "A": Placement(0, 0, Orientation.HORIZONTAL),
        "B": None,
        "C": None,
    })

    # Generate feasible neighbors
    neighbors = get_feasible_neighbors(
        original, vehicles, ship
    )

    assert len(neighbors) > 0

    # Every neighbor must satisfy hard constraints
    assert all(
        is_valid_state(n, vehicles, ship)
        for n in neighbors
    )

    # Test swap loaded A with unloaded B
    assert any(
        n.placements["A"] is None
        and n.placements["B"] == original.placements["A"]
        for n in neighbors
    )

    # Test moving unloaded B onto the ship
    assert any(
        n.placements["B"] == Placement(
            0, 2, Orientation.HORIZONTAL
        )
        for n in neighbors
    )

    # Test rotating vehicle A
    assert any(
        n.placements["A"] == Placement(
            0, 0, Orientation.VERTICAL
        )
        for n in neighbors
    )

    # Test unloading vehicle A
    assert any(
        n.placements["A"] is None
        for n in neighbors
    )

    # Original state must remain unchanged
    assert original.placements["A"] == Placement(
        0, 0, Orientation.HORIZONTAL
    )

    # Random neighbor must be feasible
    random_neighbor = get_random_feasible_neighbor(
        original, vehicles, ship,
        rng=random.Random(42)
    )

    assert random_neighbor in neighbors

    # Same seed should reproduce the same neighbor
    repeated_neighbor = get_random_feasible_neighbor(
        original, vehicles, ship,
        rng=random.Random(42)
    )

    assert random_neighbor == repeated_neighbor

    # No duplicate neighboring states
    signatures = [
        tuple(n.placements.items()) for n in neighbors
    ]

    assert len(signatures) == len(set(signatures))

    print("All neighbor tests passed")


if __name__ == "__main__":
    test_neighbor()
