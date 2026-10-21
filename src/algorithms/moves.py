"""Shared random-move sampler for SA and stochastic hill climbing."""

from random import Random

from src.core.constraints import is_valid_state
from src.core.models import Orientation, Placement, Ship, State, Vehicle


def propose_random_neighbor(
    state: State,
    vehicles: dict[str, Vehicle],
    ship: Ship,
    rng: Random,
    max_attempts: int = 200,
) -> State | None:
    """Sample one random feasible neighbor without enumerating the space.

    Move families (spec): swap two vehicles, move one vehicle to another
    coordinate (including load / unload), rotate one vehicle.
    Returns None if ``max_attempts`` random tries are all infeasible.
    """
    ids = list(vehicles)
    for _ in range(max_attempts):
        kind = rng.random()
        candidate = state.copy()

        if kind < 0.25 and len(ids) >= 2:  # swap
            a, b = rng.sample(ids, 2)
            if state.placements[a] == state.placements[b]:
                continue
            candidate.placements[a] = state.placements[b]
            candidate.placements[b] = state.placements[a]

        elif kind < 0.80:  # move / load / unload
            vid = rng.choice(ids)
            current = state.placements[vid]
            if current is not None and rng.random() < 0.15:
                candidate.placements[vid] = None  # unload
            else:
                orientation = (
                    current.orientation
                    if current is not None
                    else rng.choice(list(Orientation))
                )
                w, l = Placement(0, 0, orientation).dimension(vehicles[vid])
                if w > ship.width or l > ship.length:
                    continue
                new = Placement(
                    rng.randint(0, ship.width - w),
                    rng.randint(0, ship.length - l),
                    orientation,
                )
                if new == current:
                    continue
                candidate.placements[vid] = new

        else:  # rotate
            vid = rng.choice(ids)
            current = state.placements[vid]
            if current is None:
                continue
            flipped = (
                Orientation.VERTICAL
                if current.orientation == Orientation.HORIZONTAL
                else Orientation.HORIZONTAL
            )
            candidate.placements[vid] = Placement(current.x, current.y, flipped)

        if is_valid_state(candidate, vehicles, ship):
            return candidate
    return None
