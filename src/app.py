from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.algorithms.hill_climbing import (
    run_hill_climbing,
    run_hill_climbing_sideways,
    run_hill_climbing_stochastic,
    run_hill_climbing_random_restart,
)
from src.algorithms.simulated_annealing import run_simulated_annealing
from src.algorithms.genetic_algorithm import run_genetic_algorithm
from src.algorithms.objectives import eta_weighted_objective
from src.core.constraints import is_valid_state
from src.core.dataset_loader import load_problem
from src.core.initializer import generate_random_state
from src.core.objective import calculate_objective
from src.visualization.ship_plot import plot_ship
from src.visualization.interactive_charts import (
    optimization_figure, sa_diagnostic_figure, comparison_figure,
)
from src.visualization.interactive_ship import ship_figure, selected_vehicle_id


st.set_page_config(page_title="Ship Loading Optimizer", page_icon="🚢", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "Synthetic 10×20 · 30 Vehicles"
DATASET_PATH = PROJECT_ROOT / "data" / "synthetic_10x20"


@st.cache_data
def load_dataset(dataset_path: str):
    return load_problem(Path(dataset_path))


def show_ship(state, vehicles, ship, title):
    key = "ship_" + "".join(ch if ch.isalnum() else "_" for ch in title)
    event = st.plotly_chart(
        ship_figure(state, vehicles, ship, title),
        use_container_width=True,
        key=key,
        on_select="rerun",
        selection_mode=["points"],
        config={"displaylogo": False, "scrollZoom": False},
    )

    loaded = sorted(state.loaded_vehicle_ids())
    clicked = selected_vehicle_id(event)
    # Click a vehicle label/center, or use the selector as an accessible fallback.
    selection_key = key + "_selection"
    if clicked in loaded:
        st.session_state[selection_key] = clicked
    elif st.session_state.get(selection_key, "—") not in (["—"] + loaded):
        st.session_state[selection_key] = "—"
    chosen = st.selectbox(
        "Inspect vehicle", options=["—"] + loaded,
        key=selection_key,
        help="Hover over a vehicle for details, or select one to keep its details visible.",
    )
    if chosen != "—":
        p = state.placements[chosen]
        v = vehicles[chosen]
        w, l = p.dimension(v)
        orient = getattr(p.orientation, "value", str(p.orientation))
        st.caption(
            f"**{chosen}** · Position ({p.x}, {p.y}) · {orient} · "
            f"Footprint {w} × {l} · Weight {v.weight:g} · "
            f"Shipping Fee {v.shipping_fee:,.0f} · ETA {v.eta}"
        )


def running_best(history):
    values = []
    best = float("-inf")
    for value in history:
        best = max(best, value)
        values.append(best)
    return values


# --------------------------- Shared configuration ---------------------------
with st.sidebar:
    st.header("Configuration")
    seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
    load_prob = st.slider("Load Probability", 0.0, 1.0, 0.7, 0.1)
    generate_button = st.button("Generate Initial State", use_container_width=True)

    st.divider()
    objective_mode = st.selectbox(
        "Objective Function",
        ["Shipping Fee", "ETA-Weighted Shipping Fee"],
        key="objective_mode",
    )
    urgency_weight = 0.5
    if objective_mode == "ETA-Weighted Shipping Fee":
        urgency_weight = st.slider(
            "Urgency Weight", min_value=0.0, max_value=2.0,
            value=0.5, step=0.1, key="urgency_weight",
        )
        st.caption("Weighted objective prioritizes lower ETA; it is not actual revenue.")
        
    st.divider()
    st.caption("Optimization Parameters")
    max_iterations = st.number_input(
        "Maximum Iterations", min_value=1, max_value=10000, value=100, step=1
    )

path = DATASET_PATH
if not path.is_dir():
    st.error(f"Dataset directory not found: {path}")
    st.stop()

try:
    ship, vehicles = load_dataset(str(path))
except Exception as exc:
    st.error(f"Cannot load dataset: {exc}")
    st.stop()

objective_function = (
    partial(eta_weighted_objective, urgency_weight=float(urgency_weight))
    if objective_mode == "ETA-Weighted Shipping Fee"
    else calculate_objective
)
objective_config = (objective_mode, float(urgency_weight))
config = (DATASET_NAME, int(seed), float(load_prob))
new_state_needed = (
    "initial_state" not in st.session_state
    or generate_button
    or st.session_state.get("initial_config") != config
)

if new_state_needed:
    st.session_state.initial_state = generate_random_state(
        vehicles=vehicles, ship=ship, load_prob=float(load_prob), seed=int(seed)
    )
    st.session_state.initial_config = config
    st.session_state.pop("optimization_result", None)
    st.session_state.pop("optimization_config", None)
    st.session_state.pop("comparison_results", None)
    st.session_state.pop("comparison_config", None)

initial_state = st.session_state.initial_state
if not is_valid_state(initial_state, vehicles, ship):
    st.error("Initial state violates the ship constraints.")
    st.stop()

st.title("🚢 Ship Loading Optimizer")
st.caption(
    f"{DATASET_NAME}  ·  Ship {ship.width} × {ship.length}  ·  "
    f"Capacity {ship.max_capacity}  ·  {len(vehicles)} vehicles"
)

tab_single, tab_comparison = st.tabs(["Single Optimization", "Algorithm Comparison"])

# ----------------------------- Single algorithm -----------------------------
with tab_single:
    left_control, right_control = st.columns([1, 2], vertical_alignment="bottom")
    with left_control:
        algorithm = st.selectbox(
            "Algorithm", ["Hill Climbing",
                          "Hill Climbing (Sideways)",
                          "Hill Climbing (Stochastic)",
                          "Hill Climbing (Random Restart)",
                          "Simulated Annealing",
                          "Genetic Algorithm",
            ], key="single_algorithm"
        )
    with right_control:
        st.caption("Run one algorithm and inspect its best configuration.")
    if algorithm == "Hill Climbing (Sideways)":
        max_sideways = st.number_input(
            "Maximum Sideways Moves", min_value=0, max_value=10000,
            value=20, step=1, key="hc_max_sideways",
        )
        single_config = (config, algorithm, int(max_iterations), int(max_sideways), int(seed))

    elif algorithm == "Hill Climbing (Stochastic)":
        single_config = (config, algorithm, int(max_iterations), int(seed))

    elif algorithm == "Hill Climbing (Random Restart)":
        max_restarts = st.number_input(
            "Maximum Restarts", min_value=0, max_value=100,
            value=3, step=1, key="hc_max_restarts",
        )
        single_config = (
            config, algorithm, int(max_iterations), int(max_restarts),
            float(load_prob), int(seed),
        )
    elif algorithm == "Simulated Annealing":
        temp_col, cooling_col = st.columns(2)
        with temp_col:
            initial_temperature = st.number_input("Initial Temperature", min_value=0.01, value=100.0)
        with cooling_col:
            cooling_rate = st.slider("Cooling Rate", 0.900, 0.999, 0.995, 0.001, format="%.3f")
        single_config = (
            config, algorithm, int(max_iterations),
            float(initial_temperature), float(cooling_rate),
        )
    elif algorithm == "Genetic Algorithm":
        pop_col, mutation_col = st.columns(2)
        with pop_col:
            population_size = st.number_input(
                "Population Size", min_value=2, max_value=500, value=40, step=1
            )
        with mutation_col:
            mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.1, 0.01)
        single_config = (
            config, algorithm, int(max_iterations),
            int(population_size), float(mutation_rate),
        )
    else:
        single_config = (config, algorithm, int(max_iterations))

    single_config = (*single_config, objective_config)

    if st.button("Run Optimization", type="primary", key="run_single"):
        try:
            with st.spinner(f"Running {algorithm}..."):
                if algorithm == "Hill Climbing":
                    result = run_hill_climbing(
                        initial_state=initial_state,
                        vehicles=vehicles,
                        ship=ship,
                        max_iterations=int(max_iterations),
                        objective=objective_function, 
                    )
                elif algorithm == "Hill Climbing (Sideways)":
                    result = run_hill_climbing_sideways(
                        initial_state=initial_state,
                        vehicles=vehicles,
                        ship=ship,
                        max_sideways=int(max_sideways),
                        max_iterations=int(max_iterations),
                        seed=int(seed),
                        objective=objective_function, 
                    )
                elif algorithm == "Hill Climbing (Stochastic)":
                    result = run_hill_climbing_stochastic(
                        initial_state=initial_state,
                        vehicles=vehicles,
                        ship=ship,
                        max_iterations=int(max_iterations),
                        seed=int(seed),
                        objective=objective_function, 
                    )
                elif algorithm == "Hill Climbing (Random Restart)":
                    result = run_hill_climbing_random_restart(
                        initial_state=initial_state,
                        vehicles=vehicles,
                        ship=ship,
                        max_restarts=int(max_restarts),
                        max_iterations=int(max_iterations),
                        load_prob=float(load_prob),
                        seed=int(seed),
                        objective=objective_function, 
                    )
                elif algorithm == "Simulated Annealing":
                    result = run_simulated_annealing(
                        initial_state=initial_state,
                        vehicles=vehicles,
                        ship=ship,
                        initial_temperature=float(initial_temperature),
                        cooling_rate=float(cooling_rate),
                        max_iterations=int(max_iterations),
                        seed=int(seed),
                        objective=objective_function, 
                    )
                elif algorithm == "Genetic Algorithm":
                    result = run_genetic_algorithm(
                        initial_state=initial_state,
                        vehicles=vehicles,
                        ship=ship,
                        population_size=int(population_size),
                        max_generations=int(max_iterations),
                        mutation_rate=float(mutation_rate),
                        seed=int(seed),
                        objective=objective_function, 
                    )
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")
            st.session_state.optimization_result = result
            st.session_state.optimization_config = single_config
        except Exception as exc:
            st.session_state.pop("optimization_result", None)
            st.error(f"Optimization failed: {exc}")

    result = st.session_state.get("optimization_result")
    if st.session_state.get("optimization_config") != single_config:
        result = None

    if result is None:
        # The initial configuration is the only ship layout before optimization.
        loaded = initial_state.loaded_vehicle_ids()
        initial_fee = calculate_objective(initial_state, vehicles)
        used_weight = sum(vehicles[vid].weight for vid in loaded)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Initial Shipping Fee", f"{initial_fee:,.0f}")
        metric_cols[1].metric("Loaded Vehicles", f"{len(loaded)} / {len(vehicles)}")
        metric_cols[2].metric(
            "Weight Used", f"{used_weight:g} / {ship.max_capacity:g}"
        )
        st.subheader("Initial Ship Configuration")
        show_ship(initial_state, vehicles, ship, "Initial State")
    else:
        first_fee = calculate_objective(result.initial_state, vehicles)
        best_fee = calculate_objective(result.best_state, vehicles)
        delta = best_fee - first_fee
        improvement = f"{delta:+,.0f}"
        if first_fee > 0:
            improvement = f"{(delta / first_fee) * 100:+.1f}%"

        if objective_mode == "ETA-Weighted Shipping Fee":
            initial_obj = objective_function(result.initial_state, vehicles)
            best_obj = objective_function(result.best_state, vehicles)
            metric_cols = st.columns(4)
            metric_cols[0].metric("Initial Objective", f"{initial_obj:,.1f}")
            metric_cols[1].metric("Best Objective", f"{best_obj:,.1f}",
                                  delta=f"{best_obj - initial_obj:+,.1f}")
            metric_cols[2].metric("Actual Shipping Fee", f"{best_fee:,.0f}")
            metric_cols[3].metric("Runtime", f"{result.runtime:.3f} s")
            st.caption(f"{result.iterations} iterations / generations. Objective score includes ETA weighting.")
        else:
            metric_cols = st.columns(4)
            metric_cols[0].metric("Initial Fee", f"{first_fee:,.0f}")
            metric_cols[1].metric("Best Fee", f"{best_fee:,.0f}", delta=improvement)
            metric_cols[2].metric("Runtime", f"{result.runtime:.3f} s")
            metric_cols[3].metric("Iterations", result.iterations)

        st.subheader("Ship Layout Comparison")
        before, after = st.columns(2)
        with before:
            show_ship(result.initial_state, vehicles, ship, "Initial State")
        with after:
            show_ship(result.best_state, vehicles, ship, "Best State")

        st.subheader("Optimization Progress")
        st.plotly_chart(optimization_figure(result), use_container_width=True,
                        config={"displaylogo": False})

        if result.algorithm == "Simulated Annealing":
            st.subheader("Simulated Annealing Diagnostics")
            temp_history = result.metadata.get("temperature_history", [])
            prob_history = result.metadata.get("acceptance_probability_history", [])
            left_diag, right_diag = st.columns(2)
            with left_diag:
                if temp_history:
                    st.plotly_chart(
                        sa_diagnostic_figure(temp_history, "Temperature Cooling", "Temperature"),
                        use_container_width=True, config={"displaylogo": False},
                    )
            with right_diag:
                if prob_history:
                    st.plotly_chart(
                        sa_diagnostic_figure(prob_history, "Acceptance Probability",
                                             "Probability", probability=True),
                        use_container_width=True, config={"displaylogo": False},
                    )
            st.caption(
                f"Acceptance rate: {result.metadata.get('acceptance_rate', 0):.1%}"
            )

        if not is_valid_state(result.best_state, vehicles, ship):
            st.error("Best state violates ship constraints.")

    st.subheader("Vehicle Dataset")
    active_state = result.best_state if result is not None else initial_state
    loaded_ids = set(active_state.loaded_vehicle_ids())
    rows = [
        {
            "ID": vid,
            "Width": v.width,
            "Length": v.length,
            "Shipping Fee": v.shipping_fee,
            "Weight": v.weight,
            "ETA": v.eta,
            "Status": "Loaded" if vid in loaded_ids else "Unloaded",
        }
        for vid, v in vehicles.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Status reflects the best state when an optimization result is shown.")
    
# ---------------------------- Algorithm comparison ---------------------------
with tab_comparison:
    comparison_options = [
        "Hill Climbing",
        "Hill Climbing (Sideways)",
        "Hill Climbing (Stochastic)",
        "Hill Climbing (Random Restart)",
        "Simulated Annealing",
        "Genetic Algorithm",
    ]
    selected_algorithms = st.multiselect(
        "Algorithms to Compare",
        options=comparison_options,
        default=["Hill Climbing", "Simulated Annealing", "Genetic Algorithm"],
        key="comparison_algorithms",
    )
    st.caption("All selected algorithms use the same dataset, initial state, and objective.")

    compare_sideways = 20
    compare_restarts = 3
    if "Hill Climbing (Sideways)" in selected_algorithms:
        compare_sideways = st.number_input(
            "Comparison: Maximum Sideways Moves", min_value=0, max_value=10000,
            value=20, step=1, key="comparison_sideways",
        )
    if "Hill Climbing (Random Restart)" in selected_algorithms:
        compare_restarts = st.number_input(
            "Comparison: Maximum Restarts", min_value=0, max_value=100,
            value=3, step=1, key="comparison_restarts",
        )

    st.caption(
        "Fixed settings: SA temperature 100 / cooling 0.995; "
        "GA population 40 / mutation 0.1."
    )

    compare_config = (
        config, objective_config, int(max_iterations),
        tuple(selected_algorithms), int(compare_sideways), int(compare_restarts),
        100.0, 0.995, 40, 0.1,
    )

    if st.button("Run Comparison", type="primary", key="run_comparison"):
        if not selected_algorithms:
            st.warning("Select at least one algorithm to compare.")
        else:
            try:
                results_new = {}
                for name in selected_algorithms:
                    with st.spinner(f"Running {name}..."):
                        if name == "Hill Climbing":
                            item = run_hill_climbing(
                                initial_state=initial_state, vehicles=vehicles,
                                ship=ship, max_iterations=int(max_iterations),
                                objective=objective_function,
                            )
                        elif name == "Hill Climbing (Sideways)":
                            item = run_hill_climbing_sideways(
                                initial_state=initial_state, vehicles=vehicles,
                                ship=ship, max_sideways=int(compare_sideways),
                                max_iterations=int(max_iterations), seed=int(seed),
                                objective=objective_function,
                            )
                        elif name == "Hill Climbing (Stochastic)":
                            item = run_hill_climbing_stochastic(
                                initial_state=initial_state, vehicles=vehicles,
                                ship=ship, max_iterations=int(max_iterations),
                                seed=int(seed), objective=objective_function,
                            )
                        elif name == "Hill Climbing (Random Restart)":
                            item = run_hill_climbing_random_restart(
                                initial_state=initial_state, vehicles=vehicles,
                                ship=ship, max_restarts=int(compare_restarts),
                                max_iterations=int(max_iterations),
                                load_prob=float(load_prob), seed=int(seed),
                                objective=objective_function,
                            )
                        elif name == "Simulated Annealing":
                            item = run_simulated_annealing(
                                initial_state=initial_state, vehicles=vehicles,
                                ship=ship, initial_temperature=100.0,
                                cooling_rate=0.995,
                                max_iterations=int(max_iterations),
                                seed=int(seed), objective=objective_function,
                            )
                        elif name == "Genetic Algorithm":
                            item = run_genetic_algorithm(
                                initial_state=initial_state, vehicles=vehicles,
                                ship=ship, population_size=40,
                                max_generations=int(max_iterations),
                                mutation_rate=0.1, seed=int(seed),
                                objective=objective_function,
                            )
                        else:
                            raise ValueError(f"Unsupported algorithm: {name}")
                    results_new[name] = item

                st.session_state.comparison_results = results_new
                st.session_state.comparison_config = compare_config
            except Exception as exc:
                st.session_state.pop("comparison_results", None)
                st.session_state.pop("comparison_config", None)
                st.error(f"Comparison failed: {exc}")

    results = st.session_state.get("comparison_results")
    if st.session_state.get("comparison_config") != compare_config:
        results = None

    if results is not None:
        st.subheader("Performance Summary")
        initial_obj = objective_function(initial_state, vehicles)
        summary = []
        for name, item in results.items():
            score = objective_function(item.best_state, vehicles)
            actual_fee = calculate_objective(item.best_state, vehicles)
            summary.append({
                "Algorithm": name,
                "Initial Objective": initial_obj,
                "Best Objective": score,
                "Improvement (%)": (
                    round((score - initial_obj) / initial_obj * 100, 2)
                    if initial_obj > 0 else None
                ),
                "Actual Shipping Fee": actual_fee,
                "Runtime (s)": round(item.runtime, 4),
                "Iterations": item.iterations,
                "Valid": is_valid_state(item.best_state, vehicles, ship),
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        st.subheader("Convergence Comparison")
        st.plotly_chart(comparison_figure(results), use_container_width=True,
                        config={"displaylogo": False})
        st.caption(
            "Iteration counts are not computationally equivalent: HC scans many "
            "neighbors, SA samples candidates, and GA evaluates a population "
            "per generation. Random Restart combines multiple HC runs; "
            "compare runtime as well as scores."
        )

        st.subheader("Best Ship Configurations")
        result_items = list(results.items())
        for offset in range(0, len(result_items), 2):
            columns = st.columns(2)
            for col, (name, item) in zip(columns, result_items[offset:offset + 2]):
                with col:
                    show_ship(item.best_state, vehicles, ship, name)
                    st.caption(
                        f"{len(item.best_state.loaded_vehicle_ids())} / "
                        f"{len(vehicles)} vehicles loaded"
                    )