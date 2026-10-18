
from src.core.models import Vehicle, Ship
from src.core.initializer import generate_random_state
from src.core.constraints import is_valid_state


def test_initializer():

    ship = Ship(
        width=10,
        length=10,
        max_capacity=15,
    )

    vehicles = {
        "A": Vehicle("A", 3, 4, 60, 5, 2),
        "B": Vehicle("B", 2, 3, 40, 6, 3),
        "C": Vehicle("C", 4, 5, 90, 8, 1),
        "D": Vehicle("D", 2, 2, 30, 4, 4),
    }

    # Test 1: Random state must be feasible
    for seed in range(20):
        state = generate_random_state(
            vehicles,
            ship,
            seed=seed,
        )

        assert is_valid_state(state, vehicles, ship)

        # Every vehicle is represented
        assert set(state.placements) == set(vehicles)

    # Test 2: Same seed produces same state
    state_a = generate_random_state(
        vehicles,
        ship,
        seed=42,
    )

    state_b = generate_random_state(
        vehicles,
        ship,
        seed=42,
    )

    assert state_a == state_b

    # Test 3: Probability zero leaves all unloaded
    empty_state = generate_random_state(
        vehicles,
        ship,
        load_prob=0.0,
        seed=42,
    )

    assert len(empty_state.loaded_vehicle_ids()) == 0
    assert len(empty_state.unloaded_vehicle_ids()) == 4
    assert is_valid_state(empty_state, vehicles, ship)

    # Test 4: Capacity is never exceeded
    for seed in range(20):
        state = generate_random_state(
            vehicles,
            ship,
            load_prob=1.0,
            seed=seed,
        )

        loaded_weight = sum(
            vehicles[vehicle_id].weight
            for vehicle_id in state.loaded_vehicle_ids()
        )

        assert loaded_weight <= ship.max_capacity
        assert is_valid_state(state, vehicles, ship)

    # Test 5: Reject invalid probability
    try:
        generate_random_state(
            vehicles,
            ship,
            load_prob=1.5,
        )
        assert False, "Expected ValueError"

    except ValueError:
        pass

    print("All initializer tests passed")


if __name__ == "__main__":
    test_initializer()
