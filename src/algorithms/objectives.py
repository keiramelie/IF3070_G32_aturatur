from collections.abc import Callable

from src.core.models import State, Vehicle
from src.core.objective import calculate_objective

Objective = Callable[[State, dict[str, Vehicle]], float]


def eta_weighted_objective(
    state: State, vehicles: dict[str, Vehicle], urgency_weight: float = 0.5
) -> float:
    if urgency_weight < 0:
        raise ValueError("urgency_weight must be non-negative")
    eta_max = max((v.eta for v in vehicles.values()), default=0)
    if eta_max <= 0:
        return calculate_objective(state, vehicles)
    total = 0.0
    for vid in state.loaded_vehicle_ids():
        v = vehicles[vid]
        urgency = (eta_max - v.eta) / eta_max
        total += v.shipping_fee * (1.0 + urgency_weight * urgency)
    return total


OBJECTIVES: dict[str, Objective] = {
    "fee": calculate_objective,  # objective wajib
    "eta_weighted": eta_weighted_objective,  # bonus
}
