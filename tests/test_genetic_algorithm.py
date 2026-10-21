"""Tests for the GA implementation."""

import unittest

from src.algorithms.genetic_algorithm import (
    evaluate_population,
    initialize_population,
)
from src.core.constraints import is_valid_state
from src.core.models import Orientation, Placement, Ship, State, Vehicle


class GeneticAlgorithmPopulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vehicles = {
            "A": Vehicle("A", 2, 2, 70, 4, 1),
            "B": Vehicle("B", 1, 1, 30, 3, 2),
        }
        self.ship = Ship(4, 4, 10)
        self.initial = State({
            "A": Placement(0, 0, Orientation.HORIZONTAL),
            "B": None,
        })

    def test_population_reproducible_and_feasible(self) -> None:
        first = initialize_population(
            self.initial, self.vehicles, self.ship, population_size=8, seed=42
        )
        second = initialize_population(
            self.initial, self.vehicles, self.ship, population_size=8, seed=42
        )
        self.assertEqual(len(first), 8)
        self.assertEqual([s.placements for s in first], [s.placements for s in second])
        self.assertTrue(all(is_valid_state(s, self.vehicles, self.ship) for s in first))
        self.assertEqual(first[0].placements, self.initial.placements)
        self.assertIsNot(first[0], self.initial)
        first[0].placements["A"] = None
        self.assertIsNotNone(self.initial.placements["A"])

    def test_population_rejects_bad_configuration(self) -> None:
        with self.assertRaises(ValueError):
            initialize_population(self.initial, self.vehicles, self.ship, 1)
        with self.assertRaises(ValueError):
            initialize_population(State({"A": None}), self.vehicles, self.ship, 4)

    def test_evaluation_uses_core_objective(self) -> None:
        only_a = self.initial
        only_b = State({
            "A": None,
            "B": Placement(0, 0, Orientation.VERTICAL),
        })
        neither = State({"A": None, "B": None})
        scores, maximum, average = evaluate_population(
            [only_a, only_b, neither], self.vehicles, self.ship
        )
        self.assertEqual(scores, [70.0, 30.0, 0.0])
        self.assertEqual(maximum, 70.0)
        self.assertAlmostEqual(average, 100 / 3)

    def test_evaluation_rejects_invalid_individual(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_population([], self.vehicles, self.ship)
        with self.assertRaises(ValueError):
            evaluate_population([State({"A": None})], self.vehicles, self.ship)

        overlapping = State({
            "A": Placement(0, 0, Orientation.HORIZONTAL),
            "B": Placement(0, 0, Orientation.VERTICAL),
        })
        with self.assertRaises(ValueError):
            evaluate_population([overlapping], self.vehicles, self.ship)

        overweight_ship = Ship(4, 4, 3)
        with self.assertRaises(ValueError):
            evaluate_population([self.initial], self.vehicles, overweight_ship)


if __name__ == "__main__":
    unittest.main()