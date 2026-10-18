from src.core.models import (
    Vehicle,
    Ship,
    State,
    Placement,
    Orientation
)

def is_within_bounds(placement: Placement, vehicle: Vehicle, ship: Ship) -> bool:
    width, length = placement.dimension(vehicle)
    return (
        placement.x >= 0
        and placement.y >= 0
        and placement.x + width <= ship.width
        and placement.y + length <= ship.length
    )

def has_no_overlap(state: State, vehicles: dict[str, Vehicle]) -> bool:
    occupied_cells = set()
    for vehicle_id in state.loaded_vehicle_ids():
        placement = state.placements[vehicle_id]
        vehicle = vehicles[vehicle_id]
        width, length = placement.dimension(vehicle)
        for y in range(placement.y, placement.y+length):
            for x in range(placement.x, placement.x+width):
                cell = (x,y)
                if cell in occupied_cells:
                    return False
                occupied_cells.add(cell)
    return True

def is_within_capacity(state: State, vehicles: dict[str, Vehicle], ship: Ship) -> bool:
    total_weight = sum(vehicles[vehicle_id].weight
                       for vehicle_id in state.loaded_vehicle_ids())
    return total_weight <= ship.max_capacity

def is_valid_state(state: State, vehicles: dict[str, Vehicle], ship: Ship) -> bool:
    if set(state.placements.keys()) != set(vehicles.keys()):
        return False
    if (ship.width <= 0 or ship.length <= 0):
        return False
    for vehicle in vehicles.values():
        if (vehicle.width <= 0
            or vehicle.length <= 0
            or vehicle.weight < 0
            or vehicle.shipping_fee < 0
            or vehicle.eta < 0):
            return False
    for vehicle_id, placement in state.placements.items():
        if placement is None:
            continue
        if not isinstance(placement, Placement):
            return False
        if (type(placement.x) is not int
            or type(placement.y) is not int
            or not isinstance(placement.orientation, Orientation)):
            return False
        if not is_within_bounds(placement, vehicles[vehicle_id], ship):
            return False
    return (has_no_overlap(state, vehicles) and is_within_capacity(state, vehicles, ship))