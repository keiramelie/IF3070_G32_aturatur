
import random

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.core.models import Vehicle, Ship
from src.core.initializer import generate_random_state
from src.core.objective import calculate_objective
from src.core.constraints import is_valid_state
from src.visualization.ship_plot import plot_ship

from src.algorithms.hill_climbing import run_hill_climbing
from src.algorithms.simulated_annealing import run_simulated_annealing


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Ship Loading Optimizer",
    page_icon="🚢",
    layout="wide"
)

# SAMPLE PROBLEM DATA
ship = Ship(
    width=10,
    length=20,
    max_capacity=50
)

vehicles = {
    "V001": Vehicle("V001", 3, 4, 60, 5, 3),
    "V002": Vehicle("V002", 2, 3, 40, 4, 2),
    "V003": Vehicle("V003", 4, 5, 90, 8, 5),
    "V004": Vehicle("V004", 3, 3, 70, 6, 1),
    "V005": Vehicle("V005", 2, 4, 50, 5, 4),
    "V006": Vehicle("V006", 4, 4, 80, 7, 2),
}

# SIDEBAR
with st.sidebar:
    st.header("Configuration")

    st.subheader("Ship Information")
    st.write(f"Width: {ship.width} grid")
    st.write(f"Length: {ship.length} grid")
    st.write(f"Max Capacity: {ship.max_capacity}")

    st.divider()

    st.subheader("Initial State")

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        value=42,
        step=1
    )

    load_prob = st.slider(
        "Load Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    generate_button = st.button(
        "Generate Initial State",
        use_container_width=True,
        type="primary"
    )
        
    st.divider()
    st.subheader("Optimization Settings")

    algorithm = st.selectbox(
        "Algorithm",
        ["Hill Climbing", "Simulated Annealing"]
    )

    max_iterations = st.number_input(
        "Maximum Iterations",
        min_value=1,
        max_value=10000,
        value=100,
        step=50
    )

    if algorithm == "Simulated Annealing":
        initial_temperature = st.number_input(
            "Initial Temperature",
            min_value=0.01,
            value=100.0
        )

        cooling_rate = st.slider(
            "Cooling Rate",
            min_value=0.900,
            max_value=0.999,
            value=0.995,
            step=0.001,
            format="%.3f"
        )

    run_button = st.button(
        "Run Optimization",
        type="primary",
        use_container_width=True
    )

# INITIAL STATE
config = (int(seed), float(load_prob))

if ("initial_state" not in st.session_state
    or generate_button
    or st.session_state.get("initial_config") != config
    or st.session_state.pop("optimization_result", None)):
    st.session_state.initial_state = generate_random_state(
        vehicles=vehicles,
        ship=ship,
        load_prob=load_prob,
        seed=int(seed)
    )
    st.session_state.initial_config = config

initial_state = st.session_state.initial_state

if not is_valid_state(initial_state, vehicles, ship):
    st.error("Invalid initial state detected!")
    st.stop()

# RUN OPTIMIZATION
if run_button:
    try:
        with st.spinner(f"Running {algorithm}..."):
            if algorithm == "Hill Climbing":
                result = run_hill_climbing(
                    initial_state=initial_state,
                    vehicles=vehicles,
                    ship=ship,
                    max_iterations=int(max_iterations)
                )

            else:
                result = run_simulated_annealing(
                    initial_state=initial_state,
                    vehicles=vehicles,
                    ship=ship,
                    initial_temperature=float(initial_temperature),
                    cooling_rate=float(cooling_rate),
                    max_iterations=int(max_iterations),
                    seed=int(seed)
                )

        st.session_state.optimization_result = result

    except Exception as error:
        st.session_state.pop("optimization_result", None)
        st.error(f"Optimization failed: {error}")
        st.stop()


# DASHBOARD
st.title("🚢 Ship Loading Optimizer")
st.caption("IF3070 — Local Search Optimization")

st.divider()

initial_score = calculate_objective(initial_state, vehicles)

loaded_ids = initial_state.loaded_vehicle_ids()
unloaded_ids = initial_state.unloaded_vehicle_ids()

total_weight = sum(vehicles[vid].weight for vid in loaded_ids)

capacity_percentage = (total_weight / ship.max_capacity * 100
                       if ship.max_capacity > 0 else 0)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Shipping Fee", 
            f"Rp {initial_score:,.0f} juta")

col2.metric("Loaded Vehicles", 
            f"{len(loaded_ids)} / {len(vehicles)}")

col3.metric("Weight Used", 
            f"{total_weight:g} / {ship.max_capacity:g}")

col4.metric("Capacity Utilization", 
            f"{capacity_percentage:.1f}%")

st.divider()

# Ship visualization
st.subheader("Initial Ship Configuration")

fig = plot_ship(
    state=initial_state,
    vehicles=vehicles,
    ship=ship,
    title="Initial State"
)

st.pyplot(fig)
plt.close(fig)

st.divider()

# Vehicle information
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Loaded Vehicles")
    st.write(", ".join(loaded_ids) if loaded_ids else "None")

with col_right:
    st.subheader("Unloaded Vehicles")
    st.write(", ".join(unloaded_ids) if unloaded_ids else "None")

st.divider()

# Vehicle dataset table
st.subheader("Vehicle Dataset")

vehicle_rows = []

for vehicle_id, vehicle in vehicles.items():
    vehicle_rows.append({
        "ID": vehicle_id,
        "Width": vehicle.width,
        "Length": vehicle.length,
        "Shipping Fee": vehicle.shipping_fee,
        "Weight": vehicle.weight,
        "ETA": vehicle.eta,
        "Status": (
            "Loaded" if vehicle_id in loaded_ids else "Unloaded"
        )
    })

st.dataframe(
    pd.DataFrame(vehicle_rows),
    use_container_width=True,
    hide_index=True
)

st.info(
    "Optimization controls will be added after "
    "algorithm integration."
)


# OPTIMIZATION RESULTS
result = st.session_state.get("optimization_result")

if result is not None:

    st.divider()
    st.header(f"Optimization Results — {result.algorithm}")

    # METRICS
    initial_score = calculate_objective(result.initial_state, vehicles)

    best_score = calculate_objective(result.best_state, vehicles)

    improvement = best_score - initial_score

    improvement_pct = (
        (improvement / initial_score * 100)
        if initial_score > 0
        else None
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Initial Shipping Fee",
                f"{initial_score:,.0f}")

    col2.metric(
        "Optimized Shipping Fee",
        f"{best_score:,.0f}",
        delta=(
            f"{improvement_pct:+.1f}%"
            if improvement_pct is not None
            else f"{improvement:+,.0f}"
        )
    )

    col3.metric("Runtime",
                f"{result.runtime:.3f} s")

    col4.metric("Iterations",
                result.iterations)

    # SHIP STATE COMPARISON
    st.subheader("Ship Layout Comparison")

    left, right = st.columns(2)

    with left:
        st.markdown("**Before Optimization**")

        fig_initial = plot_ship(
            state=result.initial_state,
            vehicles=vehicles,
            ship=ship,
            title="Initial State"
        )

        st.pyplot(fig_initial, use_container_width=True)
        plt.close(fig_initial)

    with right:
        st.markdown("**After Optimization — Best State**")

        fig_best = plot_ship(
            state=result.best_state,
            vehicles=vehicles,
            ship=ship,
            title="Best State"
        )

        st.pyplot(fig_best, use_container_width=True)
        plt.close(fig_best)

    # OBJECTIVE HISTORY
    st.subheader("Optimization Progress")

    fig_history, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        range(len(result.history)),
        result.history,
        label="Current Objective",
        linewidth=2
    )

    best_history = []
    running_best = float("-inf")

    for score in result.history:
        running_best = max(running_best, score)
        best_history.append(running_best)

    ax.plot(
        range(len(best_history)),
        best_history,
        label="Best So Far",
        linestyle="--",
        linewidth=2
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Shipping Fee")
    ax.set_title("Objective Function vs Iteration")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig_history, use_container_width=True)
    plt.close(fig_history)

    # SIMULATED ANNEALING DIAGNOSTICS
    if result.algorithm == "Simulated Annealing":

        st.subheader("Simulated Annealing Diagnostics")

        temperature_history = result.metadata.get(
            "temperature_history", [])

        acceptance_history = result.metadata.get(
            "acceptance_probability_history", [])

        if temperature_history:
            fig_temp, ax_temp = plt.subplots(figsize=(10, 3))

            ax_temp.plot(temperature_history)
            ax_temp.set_title("Temperature Cooling")
            ax_temp.set_xlabel("Iteration")
            ax_temp.set_ylabel("Temperature")
            ax_temp.grid(alpha=0.3)

            st.pyplot(fig_temp, use_container_width=True)
            plt.close(fig_temp)

        if acceptance_history:
            fig_prob, ax_prob = plt.subplots(
                figsize=(10, 3)
            )

            ax_prob.plot(
                acceptance_history,
                linewidth=1,
                alpha=0.8
            )

            ax_prob.set_title("Acceptance Probability")
            ax_prob.set_xlabel("Iteration")
            ax_prob.set_ylabel("Probability")
            ax_prob.set_ylim(-0.05, 1.05)
            ax_prob.grid(alpha=0.3)

            st.pyplot(fig_prob, use_container_width=True)
            plt.close(fig_prob)

        st.metric(
            "Acceptance Rate",
            f"{result.metadata.get('acceptance_rate', 0):.1%}"
        )

    # VALIDATION
    st.divider()

    initial_valid = is_valid_state(result.initial_state, vehicles, ship)

    best_valid = is_valid_state(result.best_state, vehicles, ship)

    if initial_valid and best_valid:
        st.success(
            "Both initial and optimized states "
            "satisfy all ship constraints."
        )
    else:
        st.error("An invalid state was detected!")