from src.core.models import Vehicle, State

def calculate_objective(state: State, vehicles: dict[str, Vehicle]) -> float:
    total_fee = 0.0
    for vehicle_id in state.loaded_vehicle_ids():
        vehicle = vehicles[vehicle_id]
        total_fee += vehicle.shipping_fee
    return total_fee