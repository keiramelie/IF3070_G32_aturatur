"""Alternative objective functions (BONUS 1: objective selain ShippingFee).

Aturan bonus: tidak menambah variabel baru, jadi hanya memakai atribut yang
sudah ada pada Vehicle (shipping_fee, weight, eta, width, length) dan Ship.
Core (src/core) tidak diubah; objective wajib tetap ``calculate_objective``.

Asumsi ETA: ETA = sisa hari sampai kendaraan tiba/berangkat, sehingga ETA
KECIL = lebih mendesak (deadline dekat) dan diberi prioritas lebih tinggi.
"""

from collections.abc import Callable

from src.core.models import State, Vehicle
from src.core.objective import calculate_objective

Objective = Callable[[State, dict[str, Vehicle]], float]


def eta_weighted_objective(
    state: State, vehicles: dict[str, Vehicle], urgency_weight: float = 0.5
) -> float:
    """Total fee dengan bobot urgensi berdasarkan ETA.

    skor = sum( fee_i * (1 + w * (eta_max - eta_i) / eta_max) )

    Kendaraan ber-ETA kecil (mendesak) mendapat pengali hingga (1 + w),
    kendaraan ber-ETA terbesar tetap ber-pengali 1. Dengan w = 0 hasilnya
    identik dengan objective wajib.
    """
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
