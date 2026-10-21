import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.core.initializer import generate_random_state
from src.core.models import Ship, Vehicle
from experiments.ga_experiments import run_ga_parameter_studies, serialize_state


class GARunnerTests(unittest.TestCase):
    def setUp (self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.vehicles = {
            "A": Vehicle("A", 2, 1, 50, 2, 1),
            "B": Vehicle("B", 1, 2, 30, 1, 2),
            "C": Vehicle("C", 1, 1, 20, 1, 3),
        }
        self.ship = Ship(4, 4, 5)

    def test_initial_states_match_shared_hc_sa_seeds (self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = run_ga_parameter_studies (
                self.vehicles,
                self.ship,
                dataset_id="fixture",
                generation_values=(1, 2, 3),
                fixed_population_size=3,
                population_values=(2, 3, 4),
                fixed_generations=2,
                seeds=(17, 23, 31),
                output_dir=Path(directory) / "ga",
                load_probability=0.45,
            )
            for seed in (17, 23, 31):
                expected = generate_random_state (
                    self.vehicles, self.ship, load_prob=0.45, seed=seed
                )
                matching = [record for record in records if record["seed"] == seed]
                self.assertEqual(len(matching), 6)
                self.assertTrue(all(record["initial_state"] == serialize_state (expected) for record in matching))
                self.assertTrue(all(record["initializer_seed"] == seed for record in matching))
                self.assertTrue(all(record["load_probability"] == 0.45 for record in matching))
            with (Path(directory) / "ga" / "ga_manifest.json").open(encoding="utf-8") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["load_probability"], 0.45)

    def test_rejects_invalid_load_probability (self) -> None:
        for probability in (-0.1, 1.1, float("nan"), True):
            with self.subTest(probability=probability), tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory) / "ga"
                with self.assertRaises(ValueError):
                    run_ga_parameter_studies (
                        self.vehicles,
                        self.ship,
                        dataset_id="fixture",
                        generation_values=(1, 2, 3),
                        fixed_population_size=3,
                        population_values=(2, 3, 4),
                        fixed_generations=2,
                        seeds=(17, 23, 31),
                        output_dir=output_dir,
                        load_probability=probability,
                    )
                self.assertFalse(output_dir.exists())

    def test_cli_loads_csv_and_writes_study_outputs (self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "data"
            dataset_dir.mkdir()
            (dataset_dir / "ship.csv").write_text(
                "width,length,max_capacity\n4,4,5\n", encoding="utf-8"
            )
            (dataset_dir / "vehicles.csv").write_text(
                "vehicle_id,width,length,shipping_fee,weight,eta\n"
                "A,2,1,50,2,1\nB,1,2,30,1,2\nC,1,1,20,1,3\n", encoding="utf-8"
            )
            output_dir = root / "results"
            command = [
                sys.executable, "-m", "experiments.run_ga_experiments",
                "--dataset-dir", str(dataset_dir),
                "--output-dir", str(output_dir),
                "--generation-values", "1", "2", "3",
                "--fixed-population-size", "3",
                "--population-values", "2", "3", "4",
                "--fixed-generations", "2",
                "--seeds", "17", "23", "31",
                "--load-probability", "0.45",
            ]
            completed = subprocess.run (
                command, cwd=self.root, capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("Completed 18 GA runs", completed.stdout)
            with (output_dir / "ga_summary.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 18)
            self.assertEqual({row["load_probability"] for row in rows}, {"0.45"})
            second = subprocess.run (
                command, cwd=self.root, capture_output=True, text=True, check=False
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("FileExistsError", second.stderr)


if __name__ == "__main__":
    unittest.main ()
