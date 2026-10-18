
from src.core.models import (
    Vehicle,
    Ship,
    Placement,
    State,
    Orientation,
)
from src.core.grid import build_grid


def test_grid():
    ship = Ship(
        width=6,
        length=5,
        max_capacity=20,
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

    grid = build_grid(state, vehicles, ship)

    # Grid has 5 rows and 6 columns
    assert len(grid) == 5
    assert len(grid[0]) == 6

    # Vehicle A occupies 3 x 2 cells
    assert grid[0][0] == "A"
    assert grid[1][2] == "A"

    # Vehicle B occupies 2 x 3 cells
    assert grid[0][3] == "B"
    assert grid[2][4] == "B"

    # Empty cell
    assert grid[4][5] is None

    # Unloaded vehicle C is not present
    assert all("C" not in row for row in grid)

    # Every loaded vehicle occupies the correct area
    assert sum(cell == "A" for row in grid for cell in row) == 6
    assert sum(cell == "B" for row in grid for cell in row) == 6

    print("All grid tests passed")


if __name__ == "__main__":
    test_grid()
