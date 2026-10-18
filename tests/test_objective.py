
from src.core.models import (
    Vehicle,
    Placement,
    State,
    Orientation,
)
from src.core.objective import calculate_objective


def test_objective():
    # Sample vehicles
    vehicles = {
        "V001": Vehicle("V001", 3, 4, 60, 5, 3),
        "V002": Vehicle("V002", 2, 3, 40, 4, 2),
        "V003": Vehicle("V003", 4, 5, 90, 8, 5),
    }

    # State with two loaded vehicles
    state = State(
        placements={
            "V001": Placement(0, 0, Orientation.HORIZONTAL),
            "V002": None,
            "V003": Placement(4, 0, Orientation.VERTICAL),
        }
    )

    assert calculate_objective(state, vehicles) == 150

    # Unload V003
    new_state = state.copy()
    new_state.placements["V003"] = None

    assert calculate_objective(new_state, vehicles) == 60

    # Empty ship
    empty_state = State(
        placements={
            "V001": None,
            "V002": None,
            "V003": None,
        }
    )

    assert calculate_objective(empty_state, vehicles) == 0

    # Verify original state remains unchanged
    assert calculate_objective(state, vehicles) == 150

    print("All objective tests passed")


if __name__ == "__main__":
    test_objective()
