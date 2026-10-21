import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.core.models import Ship, Vehicle
from src.experiments.ga_experiments import (
    OUTPUT_FILES,
    calculate_dataset_fingerprint,
    run_ga_parameter_studies,
)


class GAExperimentTests(unittest.TestCase):
    def setUp (self) -> None:
        self.vehicles = {
            "A": Vehicle("A", 2, 1, 50, 2, 1),
            "B": Vehicle("B", 1, 2, 30, 1, 2),
            "C": Vehicle("C", 1, 1, 20, 1, 3),
            "D": Vehicle("D", 2, 2, 60, 3, 1),
        }
        self.ship = Ship(4, 4, 5)
        self.params = {
            "dataset_id": "fixture-v1",
            "generation_values": (1, 2, 3),
            "fixed_population_size": 3,
            "population_values": (2, 3, 4),
            "fixed_generations": 2,
            "seeds": (11, 22, 33),
            "mutation_rate": 0.1,
        }

    def run_study (self, output_dir: Path, **changes) -> list[dict]:
        parameters = self.params.copy()
        parameters.update(changes)
        return run_ga_parameter_studies (
            self.vehicles,
            self.ship,
            output_dir=output_dir,
            **parameters,
        )

    def test_writes_eighteen_runs_and_all_outputs (self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ga"
            records = self.run_study(destination)
            self.assertEqual(len(records), 18)
            self.assertTrue(all((destination / filename).exists() for filename in OUTPUT_FILES))
            with (destination / "ga_manifest.json").open(encoding="utf-8") as stream:
                manifest = json.load(stream)
            with (destination / "ga_runs.json").open(encoding="utf-8") as stream:
                on_disk = json.load(stream)
            with (destination / "ga_summary.csv").open(encoding="utf-8", newline="") as stream:
                summary = list(csv.DictReader(stream))
            with (destination / "ga_history.csv").open(encoding="utf-8", newline="") as stream:
                history = list(csv.DictReader(stream))
            self.assertEqual(manifest["number_of_runs"], 18)
            self.assertEqual(records, on_disk)
            self.assertEqual(len(summary), 18)
            self.assertEqual(len(history), sum(row["max_generations"] + 1 for row in records))
            self.assertEqual(len(set(row["run_id"] for row in records)), 18)
            self.assertEqual(sum(row["study"] == "vary_generations" for row in records), 9)
            self.assertEqual(sum(row["study"] == "vary_population" for row in records), 9)

    def test_paired_initial_states_and_seeds (self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = self.run_study(Path(directory) / "ga")
            for seed in self.params["seeds"]:
                matched = [record for record in records if record["seed"] == seed]
                self.assertEqual(len(matched), 6)
                self.assertEqual(len({json.dumps(row["initial_state"], sort_keys=True) for row in matched}), 1)
                self.assertEqual(len({row["initializer_seed"] for row in matched}), 1)
                self.assertEqual(len({row["algorithm_seed"] for row in matched}), 1)

    def test_fitness_curves_and_score_fields (self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = self.run_study(Path(directory) / "ga")
            for record in records:
                with self.subTest(run_id=record["run_id"]):
                    maximum = record["maximum_fitness_history"]
                    average = record["average_fitness_history"]
                    self.assertEqual(len(maximum), record["max_generations"] + 1)
                    self.assertEqual(len(average), len(maximum))
                    self.assertTrue(all(a <= b for a, b in zip(average, maximum)))
                    self.assertEqual(record["best_score"], max(maximum))
                    self.assertEqual(record["final_score"], maximum[-1])
                    self.assertGreaterEqual(record["best_score"], record["final_score"])
                    self.assertEqual(
                        record["fitness_evaluations"],
                        record["population_size"] * (record["max_generations"] + 1),
                    )
                    self.assertGreaterEqual(record["runtime_seconds"], 0)

    def test_reproducible_run_data_excluding_time (self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = self.run_study(Path(first_dir) / "ga")
            second = self.run_study(Path(second_dir) / "ga")
            normalized_first = [{key: value for key, value in row.items() if key != "runtime_seconds"} for row in first]
            normalized_second = [{key: value for key, value in row.items() if key != "runtime_seconds"} for row in second]
            self.assertEqual(normalized_first, normalized_second)

    def test_rejects_invalid_parameters_before_creating_outputs (self) -> None:
        changes = [
            {"generation_values": (1, 1, 3)},
            {"generation_values": (0, 2, 3)},
            {"population_values": (1, 3, 4)},
            {"population_values": (2, 2, 4)},
            {"fixed_population_size": True},
            {"fixed_generations": 0},
            {"seeds": (1, 1, 3)},
            {"seeds": (1, 2)},
            {"mutation_rate": float("nan")},
            {"dataset_id": "   "},
        ]
        for invalid in changes:
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "ga"
                with self.assertRaises(ValueError):
                    self.run_study(destination, **invalid)
                self.assertFalse(destination.exists())

    def test_does_not_overwrite_existing_outputs (self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ga"
            destination.mkdir()
            existing = destination / "ga_runs.json"
            existing.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.run_study(destination)
            self.assertEqual(existing.read_text(encoding="utf-8"), "existing")

    def test_dataset_fingerprint_changes_with_input (self) -> None:
        first = calculate_dataset_fingerprint(self.vehicles, self.ship)
        changed = self.vehicles.copy()
        changed["A"] = Vehicle("A", 2, 1, 51, 2, 1)
        second = calculate_dataset_fingerprint(changed, self.ship)
        self.assertNotEqual(first, second)
        self.assertEqual(first, calculate_dataset_fingerprint(dict(reversed(list(self.vehicles.items()))), self.ship))


if __name__ == "__main__":
    unittest.main()