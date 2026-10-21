
import random

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.core.models import Vehicle, Ship
from src.core.initializer import generate_random_state
from src.core.objective import calculate_objective
from src.core.constraints import is_valid_state
from src.visualization.ship_plot import plot_ship


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

# INITIAL STATE
config = (int(seed), float(load_prob))

if ("initial_state" not in st.session_state
    or generate_button
    or st.session_state.get("initial_config") != config):
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

col1.metric("Shipping Fee", f"Rp {initial_score:,.0f} juta")

col2.metric("Loaded Vehicles", f"{len(loaded_ids)} / {len(vehicles)}")

col3.metric("Weight Used", f"{total_weight:g} / {ship.max_capacity:g}")

col4.metric("Capacity Utilization", f"{capacity_percentage:.1f}%")

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
