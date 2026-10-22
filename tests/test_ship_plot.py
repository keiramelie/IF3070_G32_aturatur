import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.core.models import (
    Vehicle, Ship, State, Placement, Orientation
)
from src.visualization.ship_plot import plot_ship


def test_ship_plot():
    ship = Ship(
        width=6,
        length=5,
        max_capacity=20
    )

    vehicles = {
        "A": Vehicle("A", 3, 2, 60, 5, 2),
        "B": Vehicle("B", 2, 3, 40, 6, 3),
        "C": Vehicle("C", 2, 2, 30, 4, 1),
    }

    state = State({
        "A": Placement(0, 0, Orientation.HORIZONTAL),
        "B": Placement(3, 0, Orientation.VERTICAL),
        "C": None,
    })

    fig = plot_ship(
        state=state,
        vehicles=vehicles,
        ship=ship,
        title="Test Ship Layout"
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1

    ax = fig.axes[0]

    # Two loaded vehicles = two rectangle patches
    assert len(ax.patches) == 2

    assert ax.get_title() == "Test Ship Layout"

    # Check vehicle A dimensions
    rectangle_a = ax.patches[0]
    assert rectangle_a.get_width() == pytest.approx(2.8)
    assert rectangle_a.get_height() == pytest.approx(1.8)

    # Check vehicle B dimensions
    rectangle_b = ax.patches[1]
    assert rectangle_b.get_width() == pytest.approx(1.8)
    assert rectangle_b.get_height() == pytest.approx(2.8)

    plt.close(fig)

    print("All ship plot tests passed")


if __name__ == "__main__":
    test_ship_plot()
