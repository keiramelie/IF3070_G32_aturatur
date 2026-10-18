from src.core.models import Vehicle, Ship, State
from src.core.constraints import is_valid_state

def build_grid(state: State, vehicles: dict[str, Vehicle], ship: Ship) -> list[list[str | None]]:
    if not is_valid_state(state, vehicles, ship):
        raise ValueError("Invalid state, cannot build grid")
    grid = [[None for _ in range(ship.width)]
        for _ in range(ship.length)]
    for vehicle_id in state.loaded_vehicle_ids():
        placement = state.placements[vehicle_id]
        vehicle = vehicles[vehicle_id]
        width, length = placement.dimension(vehicle)
        for y in range(placement.y, placement.y+length):
            for x in range(placement.x, placement.x+width):
                grid[y][x] = vehicle_id
    return grid            