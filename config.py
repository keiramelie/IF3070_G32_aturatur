
from pathlib import Path

# Lokasi utama project
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset yang digunakan
DATASET_DIR = PROJECT_ROOT / "data" / "synthetic_10x20"

# Lokasi penyimpanan hasil eksperimen dan visualisasi
OUTPUT_DIR = PROJECT_ROOT / "output"

# Pengaturan eksperimen
TRIAL_SEEDS = [42, 43, 44]
LOAD_PROBABILITY = 0.5

# Parameter Hill Climbing
HC_MAX_ITERATIONS = 1000

# Parameter Simulated Annealing
SA_INITIAL_TEMPERATURE = 1000.0
SA_COOLING_RATE = 0.9995
SA_MIN_TEMPERATURE = 0.1
SA_MAX_ITERATIONS = 50000

# Bonus: varian Hill Climbing
HC_MAX_SIDEWAYS = 20
HC_MAX_RESTARTS = 5
SA_STUCK_CHECK_INTERVAL = 200  # cek local optimum eksak tiap N iterasi (mahal)
