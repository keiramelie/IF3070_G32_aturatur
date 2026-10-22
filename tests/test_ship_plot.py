import json

import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.core.models import (
    Vehicle, Ship, State, Placement, Orientation
)
from experiments.ga_experiments import calculate_dataset_fingerprint
from src.visualization.ship_plot import (
    draw_state,
    plot_ship,
    plot_ship_states,
    plot_state_comparison,
    unloaded_vehicle_report,
)


@pytest.fixture
def ga_layouts(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "ship.csv").write_text(
        "width,length,max_capacity\n4,4,10\n", encoding="utf-8"
    )
    (dataset / "vehicles.csv").write_text(
        "id,width,length,shipping_fee,weight,eta\n"
        "A,2,2,70,4,1\nB,1,1,30,3,2\n", encoding="utf-8"
    )
    ship = Ship(4, 4, 10.0)
    vehicles = {
        "A": Vehicle("A", 2, 2, 70.0, 4.0, 1),
        "B": Vehicle("B", 1, 1, 30.0, 3.0, 2),
    }
    final = {"A": {"x": 0, "y": 0, "orientation": "horizontal"}, "B": None}
    best = {**final, "B": {"x": 2, "y": 0, "orientation": "vertical"}}
    records = [
        {
            "run_id": f"vary_generations_{generations}_trial_1",
            "study": "vary_generations",
            "setting": generations,
            "trial": 1,
            "seed": 42,
            "population_size": 4,
            "max_generations": generations,
            "dataset_sha256": calculate_dataset_fingerprint(vehicles, ship),
            "initial_score": 0.0,
            "final_score": 70.0,
            "best_score": 100.0,
            "initial_state": {"A": None, "B": None},
            "final_state": final,
            "best_state": best,
        }
        for generations in (1, 2)
    ]
    source = tmp_path / "ga_runs.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    return dataset, ship, vehicles, records, source


@pytest.mark.parametrize("wrapped", [False, True])
def test_ga_and_local_state_formats_draw_same_vehicle_counts(ga_layouts, wrapped):
    _, ship, vehicles, records, _ = ga_layouts
    state = records[0]["final_state"]
    if wrapped:
        state = {"placements": state}
    fig, ax = plt.subplots()
    try:
        info = draw_state(ax, state, ship, vehicles, "Saved state")
        assert info == {"loaded": ["A"], "unloaded": ["B"], "weight": 4, "overlaps": []}
        assert len(ax.patches) == 2  # Deck outline plus vehicle A.
        assert "total fee tidak terangkut: 30" in unloaded_vehicle_report(state, vehicles)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("mode,score,loaded", [("final", "70", 1), ("best", "100", 2)])
def test_ga_comparison_matches_score_and_layout_when_final_is_worse(ga_layouts, mode, score, loaded):
    _, ship, vehicles, records, _ = ga_layouts
    fig = plot_state_comparison(records[0], ship, vehicles, result_state=mode)
    try:
        assert fig.axes[2].get_title() == f"{mode.capitalize()} State\nGenetic Algorithm"
        assert len(fig.axes[2].patches) == loaded + 1
        assert score in [text.get_text() for text in fig.axes[3].texts]
        assert f"{mode.capitalize()} score: {score}.0" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_ga_exports_distinct_runs_and_filtered_best_state(ga_layouts, tmp_path):
    dataset, _, _, _, source = ga_layouts
    original = source.read_bytes()
    destination = tmp_path / "plots"
    paths = plot_ship_states(dataset, states_path=source, output_dir=destination)
    assert {p.name for p in paths} == {
        "ship_states_vary_generations_1_trial_1.png",
        "ship_states_vary_generations_2_trial_1.png",
    }
    assert all(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in paths)
    report = (destination / "unloaded_vehicles.txt").read_text(encoding="utf-8")
    assert report.count("Kendaraan tidak dimuat: 1 dari 2") == 2

    best_paths = plot_ship_states(
        dataset, states_path=source, output_dir=destination, result_state="best",
        study="vary_generations", population_size=4, generations=2, trial=1,
    )
    assert [p.name for p in best_paths] == ["ship_states_vary_generations_2_trial_1_best.png"]
    assert all(p.exists() for p in paths)
    assert "Kendaraan tidak dimuat: 0 dari 2" in (
        destination / "unloaded_vehicles_best.txt"
    ).read_text(encoding="utf-8")
    assert source.read_bytes() == original


def test_hc_sa_wrapped_records_keep_existing_filenames(ga_layouts, tmp_path):
    dataset, _, _, records, _ = ga_layouts
    record = {
        key: value for key, value in records[0].items()
        if key in ("trial", "initial_state", "final_state", "best_state",
                   "initial_score", "final_score", "best_score")
    }
    record["algorithm"] = "Simulated Annealing"
    for key in ("initial_state", "final_state", "best_state"):
        record[key] = {"placements": record[key]}
    source = tmp_path / "states.json"
    source.write_text(json.dumps([record]), encoding="utf-8")
    paths = plot_ship_states(dataset, states_path=source, output_dir=tmp_path / "local")
    assert [p.name for p in paths] == ["ship_states_trial_1_simulated_annealing.png"]
    assert paths[0].exists()


def test_ga_rejects_wrong_dataset_and_empty_selection(ga_layouts, tmp_path):
    dataset, _, _, records, source = ga_layouts
    destination = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Tidak ada run"):
        plot_ship_states(dataset, states_path=source, output_dir=destination, trial=99)
    records[0]["dataset_sha256"] = "0" * 64
    source.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset CSV tidak cocok"):
        plot_ship_states(dataset, states_path=source, output_dir=destination)
    assert not destination.exists()


def test_ga_rejects_overlapping_layout_without_leaking_figure(ga_layouts):
    _, ship, vehicles, records, _ = ga_layouts
    record = {**records[0], "final_state": {
        "A": {"x": 0, "y": 0, "orientation": "horizontal"},
        "B": {"x": 0, "y": 0, "orientation": "vertical"},
    }}
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="Invalid state"):
        plot_state_comparison(record, ship, vehicles)
    assert plt.get_fignums() == before


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
