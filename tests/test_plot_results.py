import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.plot_results import run_analysis


class PlotResultsTests(unittest.TestCase):
    def setUp (self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ga_dir = self.root / "ga"
        self.local_dir = self.root / "local"
        self.ga_dir.mkdir()
        self.local_dir.mkdir()
        self._build_data()

    def write_csv (self, path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _build_data (self) -> None:
        seeds = (42, 43, 44)
        initial_placements = {"A": None}
        rows = []
        runs = []
        history = []
        for study, settings in (("vary_generations", (1, 2, 3)), ("vary_population", (2, 4, 6))):
            for setting in settings:
                population = 4 if study == "vary_generations" else setting
                generations = setting if study == "vary_generations" else 2
                for trial, seed in enumerate(seeds, start=1):
                    run_id = f"{study}_{setting}_trial_{trial}"
                    maximum = [100 + trial + i for i in range(generations + 1)]
                    average = [value - 5 for value in maximum]
                    record = {
                        "run_id": run_id,
                        "dataset_id": "fixture",
                        "dataset_sha256": "a" * 64,
                        "study": study,
                        "setting": setting,
                        "trial": trial,
                        "seed": seed,
                        "population_size": population,
                        "max_generations": generations,
                        "fitness_evaluations": population * (generations + 1),
                        "initial_score": 0,
                        "final_score": maximum[-1],
                        "best_score": maximum[-1],
                        "runtime_seconds": float(population * generations) / 100,
                    }
                    rows.append(record)
                    runs.append({**record, "initial_state": initial_placements})
                    for gen, (mx, avg) in enumerate(zip(maximum, average)):
                        history.append({
                            "run_id": run_id,
                            "generation": gen,
                            "maximum_fitness": mx,
                            "average_fitness": avg,
                        })
        manifest = {
            "dataset_id": "fixture",
            "dataset_sha256": "a" * 64,
            "number_of_runs": 18,
            "seeds": list(seeds),
        }
        (self.ga_dir / "ga_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (self.ga_dir / "ga_runs.json").write_text(json.dumps(runs), encoding="utf-8")
        self.write_csv(self.ga_dir / "ga_summary.csv", rows)
        self.write_csv(self.ga_dir / "ga_history.csv", history)
        local_rows = []
        local_states = []
        for seed in seeds:
            for algorithm in ("Hill Climbing", "Simulated Annealing", "Hill Climbing (Sideways)"):
                local_rows.append({
                    "dataset_id": "fixture",
                    "algorithm": algorithm,
                    "objective": "fee",
                    "seed": seed,
                    "initial_score": 0,
                    "best_score": 120 + seed - 42,
                    "runtime_seconds": 0.2,
                })
                local_states.append({
                    "dataset_id": "fixture",
                    "algorithm": algorithm,
                    "seed": seed,
                    "initial_state": {"placements": initial_placements},
                })
        self.write_csv(self.local_dir / "summary.csv", local_rows)
        (self.local_dir / "states.json").write_text(json.dumps(local_states), encoding="utf-8")

    def test_generates_reproducible_ga_graphs (self) -> None:
        destination = self.root / "ga_analysis"
        paths = run_analysis(self.ga_dir, destination, population_size=4, generations=2)
        self.assertEqual(len(paths), 5)
        self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))
        self.assertTrue((destination / "ga_convergence_pop4_gen2.png").exists())
        self.assertFalse((destination / "algorithm_comparison.csv").exists())

    def test_generates_comparison_from_matching_inputs (self) -> None:
        destination = self.root / "comparison"
        paths = run_analysis(
            self.ga_dir, destination, population_size=4, generations=2, local_dir=self.local_dir
        )
        self.assertEqual(len(paths), 8)
        with (destination / "algorithm_comparison.csv").open(encoding="utf-8", newline="") as stream:
            comparison = list(csv.DictReader(stream))
        self.assertEqual(len(comparison), 4)
        self.assertEqual({r["algorithm"] for r in comparison}, {
            "Genetic Algorithm", "Hill Climbing", "Simulated Annealing", "Hill Climbing (Sideways)"
        })
        self.assertTrue((destination / "algorithm_best_fitness.png").exists())
        self.assertTrue((destination / "algorithm_runtime.png").exists())

    def test_rejects_different_objectives (self) -> None:
        source = self.local_dir / "summary.csv"
        with source.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["objective"] = "eta_weighted"
        self.write_csv(source, rows)
        with self.assertRaisesRegex(ValueError, "mandatory fee objective"):
            run_analysis(
                self.ga_dir, self.root / "reject_objective", population_size=4,
                generations=2, local_dir=self.local_dir
            )

    def test_rejects_mismatched_initial_states (self) -> None:
        source = self.local_dir / "states.json"
        records = json.loads(source.read_text(encoding="utf-8"))
        records[0]["initial_state"]["placements"]["A"] = {"x": 0, "y": 0, "orientation": "horizontal"}
        source.write_text(json.dumps(records), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "initial states differ"):
            run_analysis(
                self.ga_dir, self.root / "reject_initial", population_size=4,
                generations=2, local_dir=self.local_dir
            )

    def test_rejects_missing_configuration (self) -> None:
        with self.assertRaisesRegex(ValueError, "three distinct seed trials"):
            run_analysis(self.ga_dir, self.root / "reject_configuration", population_size=7, generations=2)


if __name__ == "__main__":
    unittest.main()
