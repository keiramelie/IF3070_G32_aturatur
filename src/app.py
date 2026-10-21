from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.algorithms.hill_climbing import run_hill_climbing
from src.algorithms.simulated_annealing import run_simulated_annealing
from src.algorithms.genetic_algorithm import run_genetic_algorithm
from src.core.constraints import is_valid_state
from src.core.dataset_loader import load_problem
from src.core.initializer import generate_random_state
from src.core.objective import calculate_objective
from src.visualization.ship_plot import plot_ship


st.set_page_config(page_title="Ship Loading Optimizer", page_icon="🚢", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "Synthetic 10×20 · 30 Vehicles"
DATASET_PATH = PROJECT_ROOT / "data" / "synthetic_10x20"


@st.cache_data
def load_dataset(dataset_path: str):
    return load_problem(Path(dataset_path))


def show_ship(state, vehicles, ship, title):
    fig = plot_ship(state=state, vehicles=vehicles, ship=ship, title=title)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


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
            "Algorithm", ["Hill Climbing", "Simulated Annealing", "Genetic Algorithm"], key="single_algorithm"
        )
    with right_control:
        st.caption("Run one algorithm and inspect its best configuration.")

    if algorithm == "Simulated Annealing":
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

    if st.button("Run Optimization", type="primary", key="run_single"):
        try:
            with st.spinner(f"Running {algorithm}..."):
                if algorithm == "Hill Climbing":
                    result = run_hill_climbing(
                        initial_state=initial_state,
                        vehicles=vehicles,
                        ship=ship,
                        max_iterations=int(max_iterations),
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
                    )
                else:
                    result = run_genetic_algorithm(
                        initial_state=initial_state,
                        vehicles=vehicles,
                        ship=ship,
                        population_size=int(population_size),
                        max_generations=int(max_iterations),
                        mutation_rate=float(mutation_rate),
                        seed=int(seed),
                    )
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
        fig, ax = plt.subplots(figsize=(9, 3.5))
        if result.algorithm == "Genetic Algorithm":
            ax.plot(result.history, label="Generation Best")
            average_history = result.metadata.get("average_fitness_history", [])
            if average_history:
                ax.plot(average_history, label="Generation Average")
            ax.plot(running_best(result.history), "--", label="Best So Far")
            ax.set(xlabel="Generation", ylabel="Fitness (Shipping Fee)")
        else:
            ax.plot(result.history, label="Current Score")
            ax.plot(running_best(result.history), "--", label="Best So Far")
            ax.set(xlabel="Iteration", ylabel="Shipping Fee")
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if result.algorithm == "Simulated Annealing":
            st.subheader("Simulated Annealing Diagnostics")
            temp_history = result.metadata.get("temperature_history", [])
            prob_history = result.metadata.get("acceptance_probability_history", [])
            left_diag, right_diag = st.columns(2)
            with left_diag:
                if temp_history:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(temp_history)
                    ax.set(xlabel="Iteration", ylabel="Temperature")
                    ax.grid(alpha=0.25)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            with right_diag:
                if prob_history:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(prob_history)
                    ax.set(xlabel="Iteration", ylabel="Acceptance Probability")
                    ax.set_ylim(-0.05, 1.05)
                    ax.grid(alpha=0.25)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
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
    st.caption("HC, SA, and GA use the same dataset and initial ship configuration.")
    st.caption("Comparison settings: SA temperature 100 / cooling 0.995; GA population 40 / mutation 0.1.")
    # Shared comparison settings are intentionally fixed
    compare_config = (config, int(max_iterations), 100.0, 0.995)

    if st.button("Run Comparison", type="primary", key="run_comparison"):
        try:
            with st.spinner("Running HC, SA, and GA..."):
                hc = run_hill_climbing(
                    initial_state=initial_state,
                    vehicles=vehicles,
                    ship=ship,
                    max_iterations=int(max_iterations),
                )
                sa = run_simulated_annealing(
                    initial_state=initial_state,
                    vehicles=vehicles,
                    ship=ship,
                    initial_temperature=100.0,
                    cooling_rate=0.995,
                    max_iterations=int(max_iterations),
                    seed=int(seed),
                )
                ga = run_genetic_algorithm(
                    initial_state=initial_state,
                    vehicles=vehicles,
                    ship=ship,
                    population_size=40,
                    max_generations=int(max_iterations),
                    mutation_rate=0.1,
                    seed=int(seed),
                )

            st.session_state.comparison_results = {
                "Hill Climbing": hc,
                "Simulated Annealing": sa,
                "Genetic Algorithm": ga,
            }
            st.session_state.comparison_config = compare_config
        except Exception as exc:
            st.session_state.pop("comparison_results", None)
            st.error(f"Comparison failed: {exc}")

    results = st.session_state.get("comparison_results")
    if st.session_state.get("comparison_config") != compare_config:
        results = None

    if results is not None:
        st.subheader("Performance Summary")
        initial_fee = calculate_objective(initial_state, vehicles)
        summary = []
        for name, item in results.items():
            score = calculate_objective(item.best_state, vehicles)
            summary.append(
                {
                    "Algorithm": name,
                    "Initial Fee": initial_fee,
                    "Best Fee": score,
                    "Improvement (%)": (
                        round((score - initial_fee) / initial_fee * 100, 2)
                        if initial_fee > 0 else None
                    ),
                    "Runtime (s)": round(item.runtime, 4),
                    "Iterations": item.iterations,
                    "Valid": is_valid_state(item.best_state, vehicles, ship),
                }
            )
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        st.subheader("Convergence Comparison")
        fig, ax = plt.subplots(figsize=(9, 3.5))
        for name, item in results.items():
            ax.plot(running_best(item.history), label=name)
        ax.set(xlabel="Iteration", ylabel="Best Shipping Fee")
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "Iteration counts are not computationally equivalent: HC scans many "
            "neighbors, SA samples candidates, and GA evaluates a population "
            "per generation. Compare runtime as well as scores.")

        st.subheader("Best Ship Configurations")
        columns = st.columns(len(results))
        for col, (name, item) in zip(columns, results.items()):
            with col:
                show_ship(item.best_state, vehicles, ship, name)
                st.caption(
                    f"{len(item.best_state.loaded_vehicle_ids())} / "
                    f"{len(vehicles)} vehicles loaded"
                )
