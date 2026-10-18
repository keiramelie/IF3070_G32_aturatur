import random
from src.core.models import (
    Vehicle,
    Ship,
    State,
    Placement,
    Orientation,
)
from src.core.constraints import is_valid_state

def generate_random_state(vehicles: dict[str, Vehicle], ship: Ship, load_prob: float = 0.7, seed: int | None = None) -> State:
    if not 0.0 <= load_prob <= 1.0:
        raise ValueError("load_prob must be between 0 and 1")
    rng = random.Random(seed)
    state = State(placements={vehicle_id: None
                              for vehicle_id in vehicles})
    vehicle_ids = list(vehicles.keys())
    rng.shuffle(vehicle_ids)
    for vehicle_id in vehicle_ids:
        vehicle = vehicles[vehicle_id]
        if rng.random() > load_prob:
            continue
        possible_placements = []
        for orientation in Orientation:
            temp_placement = Placement( x=0, y=0, orientation=orientation)
            width, length = temp_placement.dimension(vehicle)
            if width > ship.width or length > ship.length:
                continue
            for y in range(ship.length - length + 1):
                for x in range(ship.width - width + 1):
                    possible_placements.append(Placement(x=x, y=y, orientation=orientation))
        rng.shuffle(possible_placements)
        for placement in possible_placements:
            candidate = state.copy()
            candidate.placements[vehicle_id] = placement
            if is_valid_state(candidate, vehicles, ship):
                state = candidate
                break
    return state