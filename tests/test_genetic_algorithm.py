"""Tests for the GA implementation."""

import random
import unittest
from unittest.mock import Mock, patch

from src.algorithms.genetic_algorithm import (
    evaluate_population,
    initialize_population,
    select_parent_index,
    crossover_one_point,
    repair_state,
    mutate_state,
    run_genetic_algorithm,
)
from src.core.constraints import is_valid_state
from src.core.models import Orientation, Placement, Ship, State, Vehicle
from src.core.objective import calculate_objective
from src.core.results import AlgorithmResult

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


class GeneticAlgorithmRepairTests (unittest.TestCase):
    def test_repair_preserves_feasible_placements_without_mutating_input (self) -> None:
        vehicles = {
            "A": Vehicle("A", 2, 2, 100, 3, 1),
            "B": Vehicle("B", 1, 1, 20, 1, 1),
        }
        ship = Ship(4, 3, 5)
        candidate = State({
            "A": Placement(0, 0, Orientation.HORIZONTAL),
            "B": Placement(3, 2, Orientation.VERTICAL),
        })
        before = candidate.placements.copy()
        repaired = repair_state(candidate, vehicles, ship, random.Random(11))
        self.assertEqual(repaired.placements, before)
        self.assertIsNot(repaired.placements, candidate.placements)
        repaired.placements["B"] = None
        self.assertEqual(candidate.placements, before)

    def test_repair_relocates_overlapping_vehicle_if_space_remains (self) -> None:
        vehicles = {
            "A": Vehicle("A", 2, 2, 100, 4, 1),
            "B": Vehicle("B", 1, 1, 10, 1, 1),
        }
        ship = Ship(4, 2, 10)
        candidate = State({
            "A": Placement(0, 0, Orientation.HORIZONTAL),
            "B": Placement(0, 0, Orientation.VERTICAL),
        })
        self.assertFalse(is_valid_state(candidate, vehicles, ship))
        result = repair_state(candidate, vehicles, ship, random.Random(9))
        self.assertTrue(is_valid_state(result, vehicles, ship))
        self.assertEqual(result.placements["A"], candidate.placements["A"])
        self.assertIsNotNone(result.placements["B"])
        self.assertNotEqual(result.placements["B"], candidate.placements["B"])

    def test_repair_rotates_vehicle_when_only_other_orientation_fits (self) -> None:
        vehicles = {
            "A": Vehicle("A", 2, 2, 100, 3, 1),
            "B": Vehicle("B", 2, 1, 20, 1, 1),
        }
        ship = Ship(3, 2, 10)
        candidate = State({
            "A": Placement(0, 0, Orientation.HORIZONTAL),
            "B": Placement(0, 0, Orientation.HORIZONTAL),
        })
        result = repair_state(candidate, vehicles, ship, random.Random(3))
        self.assertTrue(is_valid_state(result, vehicles, ship))
        self.assertEqual(result.placements["B"], Placement(2, 0, Orientation.VERTICAL))

    def test_repair_prioritizes_shipping_fee_when_deck_is_full (self) -> None:
        vehicles = {
            "high": Vehicle("high", 2, 2, 90, 3, 1),
            "low": Vehicle("low", 1, 1, 10, 1, 1),
        }
        ship = Ship(2, 2, 10)
        candidate = State({
            "high": Placement(0, 0, Orientation.HORIZONTAL),
            "low": Placement(0, 0, Orientation.VERTICAL),
        })
        result = repair_state(candidate, vehicles, ship, random.Random(11))
        self.assertTrue(is_valid_state(result, vehicles, ship))
        self.assertIsNotNone(result.placements["high"])
        self.assertIsNone(result.placements["low"])

    def test_repair_enforces_capacity_even_without_overlap (self) -> None:
        vehicles = {
            "high": Vehicle("high", 1, 1, 90, 5, 1),
            "low": Vehicle("low", 1, 1, 10, 2, 1),
        }
        ship = Ship(3, 1, 5)
        candidate = State({
            "high": Placement(0, 0, Orientation.VERTICAL),
            "low": Placement(1, 0, Orientation.VERTICAL),
        })
        result = repair_state(candidate, vehicles, ship, random.Random(3))
        self.assertTrue(is_valid_state(result, vehicles, ship))
        self.assertIsNotNone(result.placements["high"])
        self.assertIsNone(result.placements["low"])

    def test_repair_repositions_out_of_bounds_gene (self) -> None:
        vehicles = {"A": Vehicle("A", 1, 1, 30, 1, 1)}
        ship = Ship(2, 2, 10)
        candidate = State({"A": Placement(7, 4, Orientation.HORIZONTAL)})
        original = candidate.placements.copy()
        first = repair_state(candidate, vehicles, ship, random.Random(4))
        second = repair_state(candidate, vehicles, ship, random.Random(4))
        self.assertTrue(is_valid_state(first, vehicles, ship))
        self.assertIsNotNone(first.placements["A"])
        self.assertEqual(first.placements, second.placements)
        self.assertEqual(candidate.placements, original)

    def test_repair_unloads_vehicle_that_cannot_physically_fit (self) -> None:
        vehicles = {"huge": Vehicle("huge", 5, 4, 90, 2, 1)}
        ship = Ship(2, 2, 10)
        candidate = State({"huge": Placement(0, 0, Orientation.HORIZONTAL)})
        result = repair_state(candidate, vehicles, ship, random.Random(5))
        self.assertEqual(result.placements, {"huge": None})
        self.assertTrue(is_valid_state(result, vehicles, ship))

    def test_repair_rejects_malformed_chromosomes(self) -> None:
        vehicles = {"A": Vehicle("A", 1, 1, 30, 1, 1)}
        ship = Ship(2, 2, 10)
        invalid = (
            State({}),
            State({"A": None, "Z": None}),
            State({"A": "invalid"}),
            State({"A": Placement(True, 0, Orientation.HORIZONTAL)}),
            State({"A": Placement(0, 0, "horizontal")}),
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate.placements):
                with self.assertRaises(ValueError):
                    repair_state(candidate, vehicles, ship, random.Random(5))

    def test_repair_crossover_output_is_safe_for_fitness (self) -> None:
        vehicles = {
            vehicle_id: Vehicle(vehicle_id, 1, 1, 10, 1, 1)
            for vehicle_id in ("A", "B", "C", "D")
        }
        ship = Ship(5, 3, 10)
        parent_a = State({
            "A": Placement(0, 0, Orientation.HORIZONTAL),
            "B": Placement(1, 0, Orientation.HORIZONTAL),
            "C": Placement(2, 0, Orientation.HORIZONTAL),
            "D": Placement(3, 0, Orientation.HORIZONTAL),
        })
        parent_b = State({
            "A": Placement(4, 0, Orientation.HORIZONTAL),
            "B": Placement(3, 0, Orientation.HORIZONTAL),
            "C": Placement(0, 0, Orientation.HORIZONTAL),
            "D": Placement(1, 0, Orientation.HORIZONTAL),
        })
        rng = Mock(spec=random.Random)
        rng.randrange.return_value = 2
        child = crossover_one_point(parent_a, parent_b, sorted(vehicles), rng)
        self.assertFalse(is_valid_state(child, vehicles, ship))
        repaired = repair_state(child, vehicles, ship, random.Random(19))
        scores, _, _ = evaluate_population([repaired], vehicles, ship)
        self.assertGreaterEqual(scores[0], 0)
        self.assertTrue(is_valid_state(repaired, vehicles, ship))
        self.assertFalse(is_valid_state(child, vehicles, ship))


class GeneticAlgorithmMutationTests (unittest.TestCase):
    def test_swap_can_exchange_vehicle_placements (self) -> None:
        vehicles = {
            vehicle_id: Vehicle(vehicle_id, 1, 1, 10, 1, 1)
            for vehicle_id in ("A", "B")
        }
        ship = Ship(4, 2, 10)
        original = State({
            "A": Placement(0, 0, Orientation.HORIZONTAL),
            "B": Placement(2, 0, Orientation.VERTICAL),
        })
        rng = Mock(spec=random.Random)
        rng.choice.return_value = "swap"
        rng.sample.return_value = ["A", "B"]
        mutated = mutate_state(original, vehicles, ship, rng, max_attempts=1)
        self.assertEqual(mutated.placements["A"], original.placements["B"])
        self.assertEqual(mutated.placements["B"], original.placements["A"])
        self.assertTrue(is_valid_state(mutated, vehicles, ship))
        self.assertNotEqual(mutated.placements, original.placements)

    def test_rotation_changes_orientation_and_preserves_parent (self) -> None:
        vehicles = {"A": Vehicle("A", 2, 1, 50, 1, 1)}
        ship = Ship(4, 4, 10)
        original = State({"A": Placement(0, 0, Orientation.HORIZONTAL)})
        rng = Mock(spec=random.Random)
        rng.choice.side_effect = ["rotate", "A"]
        mutated = mutate_state(original, vehicles, ship, rng, max_attempts=1)
        self.assertEqual(mutated.placements["A"], Placement(0, 0, Orientation.VERTICAL))
        self.assertEqual(original.placements["A"].orientation, Orientation.HORIZONTAL)

    def test_move_can_load_an_unloaded_vehicle (self) -> None:
        vehicles = {"A": Vehicle("A", 2, 1, 50, 1, 1)}
        ship = Ship(4, 4, 10)
        original = State({"A": None})
        rng = Mock(spec=random.Random)
        rng.choice.side_effect = ["move", "A", Orientation.HORIZONTAL]
        rng.randrange.side_effect = [1, 2]
        mutated = mutate_state(original, vehicles, ship, rng, max_attempts=1)
        self.assertEqual(mutated.placements["A"], Placement(1, 2, Orientation.HORIZONTAL))
        self.assertIsNone(original.placements["A"])

    def test_move_can_unload_vehicle (self) -> None:
        vehicles = {"A": Vehicle("A", 1, 1, 20, 1, 1)}
        ship = Ship(2, 2, 10)
        original = State({"A": Placement(0, 0, Orientation.HORIZONTAL)})
        rng = Mock(spec=random.Random)
        rng.choice.side_effect = ["move", "A"]
        rng.random.return_value = 0.0
        mutated = mutate_state(original, vehicles, ship, rng, max_attempts=1)
        self.assertIsNone(mutated.placements["A"])
        self.assertIsNotNone(original.placements["A"])

    def test_mutation_returns_copy_if_no_valid_change_found (self) -> None:
        vehicles = {"A": Vehicle("A", 2, 2, 50, 1, 1)}
        ship = Ship(2, 2, 10)
        original = State({"A": Placement(0, 0, Orientation.HORIZONTAL)})
        rng = Mock(spec=random.Random)
        rng.choice.side_effect = ["move", "A"]
        rng.random.return_value = 0.9
        rng.randrange.side_effect = [0, 0]
        mutated = mutate_state(original, vehicles, ship, rng, max_attempts=1)
        self.assertEqual(mutated.placements, original.placements)
        self.assertIsNot(mutated, original)

    def test_mutation_rejects_infeasible_initial_and_bad_budget (self) -> None:
        vehicles = {"A": Vehicle("A", 1, 1, 20, 1, 1)}
        ship = Ship(2, 2, 10)
        invalid_state = State({"A": Placement(3, 0, Orientation.HORIZONTAL)})
        with self.assertRaises(ValueError):
            mutate_state(invalid_state, vehicles, ship, random.Random(2))
        for attempts in (0, -1, True, 1.5):
            with self.subTest(attempts=attempts):
                with self.assertRaises(ValueError):
                    mutate_state(State({"A": None}), vehicles, ship,
                                 random.Random(2), max_attempts=attempts)

    def test_mutation_reproducible_and_feasible_over_many_seeds (self) -> None:
        vehicles = {
            "A": Vehicle("A", 2, 1, 50, 1, 1),
            "B": Vehicle("B", 1, 1, 20, 1, 1),
            "C": Vehicle("C", 1, 1, 10, 1, 1),
        }
        ship = Ship(4, 4, 10)
        initial = State({
            "A": Placement(0, 0, Orientation.HORIZONTAL),
            "B": Placement(3, 0, Orientation.VERTICAL),
            "C": None,
        })
        initial_snapshot = initial.placements.copy()
        for seed in range(60):
            first = mutate_state(initial, vehicles, ship, random.Random(seed))
            second = mutate_state(initial, vehicles, ship, random.Random(seed))
            self.assertEqual(first.placements, second.placements)
            self.assertTrue(is_valid_state(first, vehicles, ship))
            self.assertEqual(initial.placements, initial_snapshot)

    def test_mutation_on_empty_dataset_returns_a_copy (self) -> None:
        initial = State({})
        ship = Ship(2, 2, 0)
        result = mutate_state(initial, {}, ship, random.Random(1))


class GeneticAlgorithmDriverTests(unittest.TestCase):
    def setUp (self) -> None:
        self.vehicles = {
            "A": Vehicle("A", 2, 1, 60, 2, 1),
            "B": Vehicle("B", 1, 2, 40, 2, 2),
            "C": Vehicle("C", 1, 1, 25, 1, 3),
            "D": Vehicle("D", 1, 1, 15, 1, 4),
        }
        self.ship = Ship(5, 4, 5)
        self.initial_state = State({
            "A": Placement(0, 0, Orientation.HORIZONTAL),
            "B": Placement(2, 0, Orientation.VERTICAL),
            "C": None,
            "D": Placement(4, 0, Orientation.HORIZONTAL),
        })
        self.assertTrue(is_valid_state(self.initial_state, self.vehicles, self.ship))

    def test_result_contract_and_fitness_histories (self) -> None:
        before = self.initial_state.placements.copy()
        result = run_genetic_algorithm(
            self.initial_state, self.vehicles, self.ship,
            population_size=7, max_generations=5, mutation_rate=0.3, seed=42,
        )
        self.assertIsInstance(result, AlgorithmResult)
        self.assertEqual(result.algorithm, "Genetic Algorithm")
        self.assertEqual(result.iterations, 5)
        self.assertEqual(len(result.history), 6)
        self.assertEqual(len(result.metadata["average_fitness_history"]), 6)
        self.assertTrue(is_valid_state(result.initial_state, self.vehicles, self.ship))
        self.assertTrue(is_valid_state(result.final_state, self.vehicles, self.ship))
        self.assertTrue(is_valid_state(result.best_state, self.vehicles, self.ship))
        self.assertEqual(before, self.initial_state.placements)
        self.assertIsNot(result.initial_state, self.initial_state)
        self.assertEqual(result.initial_state.placements, before)
        self.assertEqual(result.best_score, calculate_objective(result.best_state, self.vehicles))
        self.assertEqual(result.best_score, max(result.history))
        self.assertEqual(result.history[-1], calculate_objective(result.final_state, self.vehicles))
        self.assertEqual(result.metadata["final_population_best_score"], result.history[-1])
        self.assertEqual(result.metadata["population_size"], 7)
        self.assertEqual(result.metadata["fitness_evaluations"], 7 * 6)
        self.assertTrue(all(
            average <= maximum
            for average, maximum in zip(result.metadata["average_fitness_history"], result.history)
        ))
        self.assertGreaterEqual(result.runtime, 0)

    def test_fixed_seed_reproducibility (self) -> None:
        args = (self.initial_state, self.vehicles, self.ship)
        first = run_genetic_algorithm(*args, population_size=9, max_generations=6, mutation_rate=0.4, seed=79)
        second = run_genetic_algorithm(*args, population_size=9, max_generations=6, mutation_rate=0.4, seed=79)
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.metadata, second.metadata)
        self.assertEqual(first.final_state.placements, second.final_state.placements)
        self.assertEqual(first.best_state.placements, second.best_state.placements)
        self.assertEqual(first.best_score, second.best_score)

    def test_zero_generations_returns_best_of_initial_population (self) -> None:
        result = run_genetic_algorithm(
            self.initial_state, self.vehicles, self.ship,
            population_size=5, max_generations=0, mutation_rate=0.1, seed=8,
        )
        self.assertEqual(result.iterations, 0)
        self.assertEqual(len(result.history), 1)
        self.assertEqual(len(result.metadata["average_fitness_history"]), 1)
        self.assertEqual(result.best_score, result.history[0])
        self.assertEqual(result.best_state.placements, result.final_state.placements)
        self.assertGreaterEqual(result.best_score, calculate_objective(self.initial_state, self.vehicles))
        self.assertEqual(result.metadata["best_generation"], 0)

    def test_pure_replacement_can_lose_best_in_final_population (self) -> None:
        vehicles = {"A": Vehicle("A", 1, 1, 100, 1, 0)}
        ship = Ship(1, 1, 1)
        loaded = State({"A": Placement(0, 0, Orientation.HORIZONTAL)})
        unloaded = State({"A": None})
        with patch(
            "src.algorithms.genetic_algorithm.initialize_population",
            return_value=[loaded.copy(), unloaded.copy()],
        ), patch(
            "src.algorithms.genetic_algorithm.select_parent_index",
            return_value=1,
        ):
            result = run_genetic_algorithm(
                loaded, vehicles, ship, population_size=2,
                max_generations=1, mutation_rate=0, seed=1,
            )
        self.assertEqual(result.history, [100.0, 0.0])
        self.assertEqual(result.best_score, 100.0)
        self.assertEqual(result.best_state.placements, loaded.placements)
        self.assertEqual(result.final_state.placements, unloaded.placements)
        self.assertEqual(result.metadata["best_generation"], 0)

    def test_mutation_rate_controls_mutation_calls (self) -> None:
        with patch("src.algorithms.genetic_algorithm.mutate_state") as mutation:
            run_genetic_algorithm(
                self.initial_state, self.vehicles, self.ship,
                population_size=4, max_generations=3, mutation_rate=0, seed=9,
            )
            mutation.assert_not_called()
        with patch(
            "src.algorithms.genetic_algorithm.mutate_state",
            side_effect=lambda state, vehicles, ship, rng: state.copy(),
        ) as mutation:
            run_genetic_algorithm(
                self.initial_state, self.vehicles, self.ship,
                population_size=4, max_generations=3, mutation_rate=1, seed=9,
            )
            self.assertEqual(mutation.call_count, 12)

    def test_rejects_invalid_parameter_values (self) -> None:
        cases = [
            {"population_size": 1},
            {"population_size": True},
            {"population_size": 2.5},
            {"max_generations": -1},
            {"max_generations": True},
            {"max_generations": 1.5},
            {"mutation_rate": -0.1},
            {"mutation_rate": 1.1},
            {"mutation_rate": True},
            {"mutation_rate": float("nan")},
            {"mutation_rate": float("inf")},
            {"mutation_rate": "0.5"},
        ]
        for case in cases:
            with self.subTest(case=case):
                arguments = {"population_size": 4, "max_generations": 0, "mutation_rate": 0.1}
                arguments.update(case)
                with self.assertRaises(ValueError):
                    run_genetic_algorithm(
                        self.initial_state, self.vehicles, self.ship,
                        **arguments,
                    )

    def test_global_rng_is_unchanged_by_seeded_run (self) -> None:
        before = random.getstate()
        run_genetic_algorithm(
            self.initial_state, self.vehicles, self.ship,
            population_size=4, max_generations=3, mutation_rate=0.5, seed=777,
        )
        self.assertEqual(random.getstate(), before)

    def test_zero_vehicles (self) -> None:
        result = run_genetic_algorithm(
            State({}), {}, Ship(1, 1, 1),
            population_size=2, max_generations=3, seed=99,
        )
        self.assertEqual(result.history, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(result.metadata["average_fitness_history"], result.history)
        self.assertTrue(is_valid_state(result.best_state, {}, Ship(1, 1, 1)))


if __name__ == "__main__":
    unittest.main()