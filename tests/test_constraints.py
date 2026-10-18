
from src.core.models import (
    Vehicle,
    Ship,
    Placement,
    State,
    Orientation,
)
from src.core.constraints import (
    is_within_bounds,
    has_no_overlap,
    is_within_capacity,
    is_valid_state,
)


def test_constraints():
    ship = Ship(
        width=10,
        length=10,
        max_capacity=15,
    )

    vehicles = {
        "A": Vehicle("A", 3, 4, 60, 5, 2),
        "B": Vehicle("B", 2, 3, 40, 6, 3),
        "C": Vehicle("C", 4, 5, 90, 8, 1),
    }

    # A horizontal occupies 4 x 3, B horizontal occupies 3 x 2
    valid_state = State({
        "A": Placement(0, 0, Orientation.HORIZONTAL),
        "B": Placement(4, 0, Orientation.HORIZONTAL),
        "C": None,
    })

    assert is_valid_state(valid_state, vehicles, ship)
    assert has_no_overlap(valid_state, vehicles)
    assert is_within_capacity(valid_state, vehicles, ship)

    # A and B overlap
    overlap_state = State({
        "A": Placement(0, 0, Orientation.HORIZONTAL),
        "B": Placement(3, 1, Orientation.HORIZONTAL),
        "C": None,
    })
    assert not has_no_overlap(overlap_state, vehicles)
    assert not is_valid_state(overlap_state, vehicles, ship)

    # B exceeds the right boundary
    boundary_state = State({
        "A": None,
        "B": Placement(8, 0, Orientation.HORIZONTAL),
        "C": None,
    })
    assert not is_within_bounds(
        boundary_state.placements["B"],
        vehicles["B"],
        ship,
    )
    assert not is_valid_state(boundary_state, vehicles, ship)

    # Total weight exceeds capacity: 5 + 6 + 8 = 19 > 15
    overweight_state = State({
        "A": Placement(0, 0, Orientation.HORIZONTAL),
        "B": Placement(4, 0, Orientation.HORIZONTAL),
        "C": Placement(0, 4, Orientation.HORIZONTAL),
    })
    assert not is_within_capacity(overweight_state, vehicles, ship)
    assert not is_valid_state(overweight_state, vehicles, ship)

    # Missing vehicle entry
    incomplete_state = State({
        "A": Placement(0, 0, Orientation.HORIZONTAL),
        "B": None,
    })
    assert not is_valid_state(incomplete_state, vehicles, ship)

    # Unloaded vehicles are valid
    empty_state = State({
        "A": None,
        "B": None,
        "C": None,
    })
    assert is_valid_state(empty_state, vehicles, ship)

    print("All constraint tests passed")


if __name__ == "__main__":
    test_constraints()
