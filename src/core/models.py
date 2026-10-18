from dataclasses import dataclass, field
from enum import Enum

class Orientation(str,Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

@dataclass(frozen=True)
class Vehicle:
    id: str
    width: int
    length: int
    shipping_fee: float
    weight: float
    eta: int

@dataclass(frozen=True)
class Ship:
    width: int
    length: int
    max_capacity: float

@dataclass(frozen=True)
class Placement:
    x: int
    y: int
    orientation: Orientation

    def dimension(self, vehicle: Vehicle) -> tuple[int, int]:
        long = max(vehicle.width, vehicle.length)
        short = min(vehicle.width, vehicle.length)
        if self.orientation == Orientation.HORIZONTAL:
            return long, short
        return short, long
    
@dataclass
class State:
    placements: dict[str, Placement | None] = field(default_factory=dict)

    def copy(self) -> "State":
        return State(placements=self.placements.copy())
    
    def loaded_vehicle_ids(self) -> list[str]:
        return [vehicle_id
                for vehicle_id, placement in self.placements.items()
                if placement is not None]
    
    def unloaded_vehicle_ids(self) -> list[str]:
        return [vehicle_id
                for vehicle_id, placement in self.placements.items()
                if placement is None]