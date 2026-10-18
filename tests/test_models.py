from src.core.models import (
    Vehicle,
    Ship,
    Placement,
    State,
    Orientation,
)

def test_models():
    vehicle = Vehicle(
        id="V001",
        width=3,
        length=4,
        shipping_fee=60,
        weight=5,
        eta=3,
    )

    ship = Ship(
        width=10,
        length=20,
        max_capacity=500,
    )

    placement = Placement(
        x=0,
        y=0,
        orientation=Orientation.HORIZONTAL,
    )

    state = State(
        placements={
            "V001": placement,
            "V002": None,
        }
    )

    assert vehicle.shipping_fee == 60
    assert ship.max_capacity == 500
    assert placement.dimension(vehicle) == (4,3)

    rotated_placement = Placement(
        x=0,
        y=0,
        orientation=Orientation.VERTICAL
    )
    assert rotated_placement.dimension(vehicle) == (3,4)

    assert state.loaded_vehicle_ids() == ["V001"]
    assert state.unloaded_vehicle_ids() == ["V002"]

    coppied_state = state.copy()
    coppied_state.placements["V001"] = None
    assert state.placements["V001"] is not None
    assert coppied_state.placements["V001"] is None

    print("All model test passed")

if __name__ == "__main__":
    test_models()