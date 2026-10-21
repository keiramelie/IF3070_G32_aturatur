"""Tests for the GA implementation."""

import random
import unittest
from unittest.mock import Mock 

from src.algorithms.genetic_algorithm import (
    evaluate_population,
    initialize_population,
    select_parent_index,
    crossover_one_point,
)
from src.core.constraints import is_valid_state
from src.core.models import Orientation, Placement, Ship, State, Vehicle


class GeneticAlgorithmPopulationTests (unittest.TestCase):
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

    def test_population_reproducible_and_feasible (self) -> None:
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

    def test_population_rejects_bad_configuration (self) -> None:
        with self.assertRaises(ValueError):
            initialize_population(self.initial, self.vehicles, self.ship, 1)
        with self.assertRaises(ValueError):
            initialize_population(State({"A": None}), self.vehicles, self.ship, 4)

    def test_evaluation_uses_core_objective (self) -> None:
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

    def test_evaluation_rejects_invalid_individual (self) -> None:
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


class GeneticAlgorithmOperatorsTests (unittest.TestCase):
    def setUp (self) -> None:
        self.vehicles = {
            vehicle_id: Vehicle(vehicle_id, 1, 1, 10, 1, 1)
            for vehicle_id in ("A", "B", "C", "D")
        }
        self.ship = Ship(5, 3, 10)
        horizontal = Orientation.HORIZONTAL
        self.parent_a = State({
            "A": Placement(0, 0, horizontal),
            "B": Placement(1, 0, horizontal),
            "C": Placement(2, 0, horizontal),
            "D": Placement(3, 0, horizontal),
        })
        self.parent_b = State({
            "A": Placement(4, 0, horizontal),
            "B": Placement(3, 0, horizontal),
            "C": Placement(0, 0, horizontal),
            "D": Placement(1, 0, horizontal),
        })
        self.assertTrue(is_valid_state(self.parent_a, self.vehicles, self.ship))
        self.assertTrue(is_valid_state(self.parent_b, self.vehicles, self.ship))

    def test_roulette_selection_respects_zero_weights (self) -> None:
        rng = random.Random(11)
        self.assertTrue(
            all(select_parent_index([0.0, 70.0, 0.0], rng) == 1 for _ in range(100))
        )

    def test_roulette_uses_fitness_proportional_weights (self) -> None:
        rng = Mock(spec=random.Random)
        rng.choices.return_value = [2]
        selected = select_parent_index([1.0, 3.0, 6.0], rng)
        self.assertEqual(selected, 2)
        rng.choices.assert_called_once_with(
            range(3), weights=[1.0, 3.0, 6.0], k=1
        )

    def test_roulette_selection_is_reproducible (self) -> None:
        left = random.Random(42)
        right = random.Random(42)
        weights = [10.0, 30.0, 60.0]
        self.assertEqual(
            [select_parent_index(weights, left) for _ in range(25)],
            [select_parent_index(weights, right) for _ in range(25)],
        )

    def test_roulette_falls_back_to_uniform_for_all_zero (self) -> None:
        left = random.Random(43)
        right = random.Random(43)
        indices_a = [select_parent_index([0.0] * 4, left) for _ in range(100)]
        indices_b = [select_parent_index([0.0] * 4, right) for _ in range(100)]
        self.assertEqual(indices_a, indices_b)
        self.assertEqual(set(indices_a), {0, 1, 2, 3})

    def test_roulette_rejects_invalid_fitness (self) -> None:
        rng = random.Random(1)
        for weights in ([], [1.0, -2.0], [float('nan')],
                        [float('inf')], [True, 1.0]):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    select_parent_index(weights, rng)

    def test_one_point_crossover_preserves_parents_and_split (self) -> None:
        rng = Mock(spec=random.Random)
        rng.randrange.return_value = 2
        before_a = self.parent_a.placements.copy()
        before_b = self.parent_b.placements.copy()
        child = crossover_one_point(
            self.parent_a, self.parent_b, ["A", "B", "C", "D"], rng
        )

        rng.randrange.assert_called_once_with(1, 4)
        self.assertEqual(
            child.placements,
            {
                "A": before_a["A"], "B": before_a["B"],
                "C": before_b["C"], "D": before_b["D"],
            },
        )
        self.assertEqual(self.parent_a.placements, before_a)
        self.assertEqual(self.parent_b.placements, before_b)
        self.assertIsNot(child, self.parent_a)
        self.assertIsNot(child.placements, self.parent_a.placements)
        self.assertFalse(is_valid_state(child, self.vehicles, self.ship))

    def test_one_point_crossover_is_reproducible (self) -> None:
        ids = tuple(sorted(self.vehicles))
        child_a = crossover_one_point(
            self.parent_a, self.parent_b, ids, random.Random(19)
        )
        child_b = crossover_one_point(
            self.parent_a, self.parent_b, ids, random.Random(19)
        )
        self.assertEqual(child_a.placements, child_b.placements)

    def test_one_point_crossover_rejects_incompatible_chromosomes (self) -> None:
        rng = random.Random(1)
        with self.assertRaises(ValueError):
            crossover_one_point(
                self.parent_a, self.parent_b, ["A", "B", "B", "D"], rng
            )
        with self.assertRaises(ValueError):
            crossover_one_point(
                self.parent_a, self.parent_b, ["A", "B", "C"], rng
            )

    def test_single_gene_crossover_clones_first_parent (self) -> None:
        parent_a = State({"A": None})
        parent_b = State({"A": Placement(0, 0, Orientation.VERTICAL)})
        child = crossover_one_point(
            parent_a, parent_b, ["A"], random.Random(5)
        )
        self.assertEqual(child.placements, parent_a.placements)
        self.assertIsNot(child, parent_a)
        child.placements["A"] = parent_b.placements["A"]
        self.assertIsNone(parent_a.placements["A"])


if __name__ == "__main__":
    unittest.main()