import random
from collections.abc import Iterator
from src.core.models import (Vehicle, Ship, State, Placement, Orientation)
from src.core.constraints import is_valid_state

def generate_candidates(state: State, vehicles: dict[str, Vehicle], ship: Ship) -> Iterator[State]:
    ids = list(vehicles)
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = ids[i], ids[j]
            if state.placements[a] == state.placements[b]:
                continue
            candidate = state.copy()
            candidate.placements[a], candidate.placements[b] = (state.placements[b], state.placements[a])
            yield candidate

    for vehicle_id in ids:
        current = state.placements[vehicle_id]
        vehicle = vehicles[vehicle_id]
        if current is not None:
            candidate = state.copy()
            candidate.placements[vehicle_id] = None
            yield candidate
            orientations = (current.orientation,)
        else:
            orientations = tuple(Orientation)
        for orientation in orientations:
            reference = Placement(0,0,orientation)
            width, length = reference.dimension(vehicle)
            for y in range(ship.length - length + 1):
                for x in range(ship.width - width + 1):
                    new_placement = Placement(x,y,orientation)
                    if new_placement == current:
                        continue
                    candidate = state.copy()
                    candidate.placements[vehicle_id] = new_placement
                    yield candidate

    for vehicle_id in ids:
        current = state.placements[vehicle_id]
        if current is None:
            continue
        new_orientation = (Orientation.VERTICAL
                           if current.orientation == Orientation.HORIZONTAL
                           else Orientation.HORIZONTAL)
        candidate = state.copy()
        candidate.placements[vehicle_id] = Placement(current.x, current.y, new_orientation)
        yield candidate

def get_feasible_neighbors(state: State, vehicles: dict[str, Vehicle], ship: Ship) -> list[State]:
    if not is_valid_state(state, vehicles, ship):
        raise ValueError("invalid initial state")
    
    neighbors = []
    seen = set()
    for candidate in generate_candidates(state, vehicles, ship):
        if not is_valid_state(candidate, vehicles, ship):
            continue
        signature = tuple(candidate.placements[v] for v in vehicles)
        if signature in seen:
            continue
        seen.add(signature)
        neighbors.append(candidate)
    return neighbors

def get_random_feasible_neighbor(state: State, vehicles: dict[str, Vehicle], ship: Ship, rng: random.Random | None = None) -> State | None:
    if not is_valid_state(state, vehicles, ship):
        raise ValueError("invalid initial state")
    
    if rng is None:
        rng = random.Random()
    chosen = None
    count = 0
    seen = set()
    for candidate in generate_candidates(state, vehicles, ship):
        if not is_valid_state(candidate, vehicles, ship):
            continue
        signature = tuple(candidate.placements[v] for v in vehicles)
        if signature in seen:
            continue
        seen.add(signature)
        count += 1
        if rng.randrange(count) == 0:
            chosen = candidate
    return chosen