import argparse
from pathlib import Path

from config import DATASET_DIR, LOAD_PROBABILITY, OUTPUT_DIR, TRIAL_SEEDS
from experiments.ga_experiments import run_ga_parameter_studies
from src.core.dataset_loader import load_problem


def build_parser () -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible GA parameter studies using the shared dataset loader."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path(DATASET_DIR))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--generation-values", type=int, nargs=3, required=True)
    parser.add_argument("--fixed-population-size", type=int, required=True)
    parser.add_argument("--population-values", type=int, nargs=3, required=True)
    parser.add_argument("--fixed-generations", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, default=tuple(TRIAL_SEEDS))
    parser.add_argument("--load-probability", type=float, default=LOAD_PROBABILITY)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    return parser


def main () -> None:
    args = build_parser ().parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(OUTPUT_DIR) / "ga" / dataset_dir.name
    )
    ship, vehicles = load_problem (dataset_dir)
    records = run_ga_parameter_studies (
        vehicles,
        ship,
        dataset_id=dataset_dir.name,
        generation_values=args.generation_values,
        fixed_population_size=args.fixed_population_size,
        population_values=args.population_values,
        fixed_generations=args.fixed_generations,
        seeds=args.seeds,
        output_dir=output_dir,
        mutation_rate=args.mutation_rate,
        load_probability=args.load_probability,
    )
    print(f"Completed {len(records)} GA runs using {len(vehicles)} vehicles")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main ()
