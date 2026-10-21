
import matplotlib.pyplot as plt

from src.core.models import (
    Vehicle, Ship, State, Placement, Orientation
)
from src.visualization.ship_plot import plot_ship


ship = Ship(width=8, length=10, max_capacity=30)

vehicles = {
    "A": Vehicle("A", 3, 4, 60, 5, 2),
    "B": Vehicle("B", 2, 3, 40, 6, 3),
    "C": Vehicle("C", 3, 2, 50, 4, 1),
    "D": Vehicle("D", 2, 2, 30, 3, 4),
}

state = State({
    "A": Placement(0, 0, Orientation.HORIZONTAL),
    "B": Placement(4, 0, Orientation.VERTICAL),
    "C": Placement(0, 4, Orientation.HORIZONTAL),
    "D": None,
})

fig = plot_ship(state, vehicles, ship, "Preview Ship Layout")

plt.show()