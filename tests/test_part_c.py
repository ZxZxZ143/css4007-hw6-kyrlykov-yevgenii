"""Checks for synchronous updates, convergence safeguards and simulation limits."""

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

from grid_world import GridWorld, load_config
from part_c import check_rules, evaluate_policy, simulate, solve


class SolverTests(unittest.TestCase):
    def solve_quietly(self, world, folder):
        with contextlib.redirect_stdout(io.StringIO()):
            return solve(world, Path(folder))

    def test_both_maps_converge_and_pass_model_checks(self):
        for name, iterations in (("paper_map", 12), ("extended_map", 11)):
            with self.subTest(map=name), tempfile.TemporaryDirectory() as folder:
                world = GridWorld(load_config(), name)
                values, policy, rounds, info = self.solve_quietly(world, folder)
                checks, _ = check_rules(world, values, policy, rounds)
                self.assertTrue(all(row["passed"] for row in checks))
                self.assertEqual(info["iterations"], iterations)
                self.assertTrue(info["converged"])
                self.assertLess(info["final_delta"], 1e-6)
                self.assertEqual(len(list(Path(folder).joinpath("iterations.txt").read_text(
                    encoding="utf-8").split("delta="))) - 1, iterations + 1)

    def test_one_iteration_cap_reports_nonconvergence(self):
        config = load_config()
        config["solver"]["max_iterations"] = 1
        with tempfile.TemporaryDirectory() as folder:
            _, _, rounds, info = self.solve_quietly(GridWorld(config, "paper_map"), folder)
        self.assertFalse(info["converged"])
        self.assertEqual(info["iterations"], 1)
        # A synchronous first update must not use the newly calculated value 10.
        self.assertEqual(rounds[1][(0, 1)], -1)

    def test_reward_objective_differs_from_goal_probability(self):
        world = GridWorld(load_config(), "paper_map")
        with tempfile.TemporaryDirectory() as folder:
            _, policy, _, _ = self.solve_quietly(world, folder)
        self.assertEqual(policy[world.start], "right")
        reward_optimal = evaluate_policy(world, policy)
        start = next(row for row in reward_optimal if (row["row"], row["column"]) == (3, 1))
        self.assertAlmostEqual(start["goal_probability"], 8 / 9)
        safe_policy = policy.copy()
        safe_policy[world.start] = "up"
        safe = next(row for row in evaluate_policy(world, safe_policy)
                    if (row["row"], row["column"]) == (3, 1))
        self.assertAlmostEqual(safe["goal_probability"], 1)
        self.assertGreater(start["value"], safe["value"])

    def test_floor_rules_can_be_extended_without_solver_changes(self):
        config = copy.deepcopy(load_config())
        config["tiles"]["X"] = {"reward": -7, "movement": "deterministic"}
        config["paper_map"][0] = ".X.G"
        world = GridWorld(config, "paper_map")
        self.assertEqual(world.transitions((0, 0), "right"), [(1.0, (0, 1), -7.0)])
        self.assertEqual(world.transitions((0, 1), "down"), [(1.0, (0, 1), -7.0)])
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, info = self.solve_quietly(world, folder)
        self.assertTrue(info["converged"])

    def test_goal_on_step_100_is_terminal_not_time_limit(self):
        config = load_config()
        config["paper_map"] = ["S" + "." * 99 + "G"]
        config["simulation"]["batch_seeds"] = [11]
        config["simulation"]["episodes_per_batch"] = 1
        world = GridWorld(config, "paper_map")
        policy = {state: "right" for state in world.nonterminal}
        with tempfile.TemporaryDirectory() as folder:
            episodes, _ = simulate(world, policy, Path(folder))
        self.assertEqual((episodes[0]["steps"], episodes[0]["outcome"]), (100, "goal"))
        expected = sum(-0.9 ** i for i in range(99)) + 10 * 0.9 ** 99
        self.assertAlmostEqual(episodes[0]["discounted_return"], expected)

    def test_nonterminal_step_100_is_time_limit(self):
        config = load_config()
        config["simulation"]["batch_seeds"] = [11]
        config["simulation"]["episodes_per_batch"] = 1
        world = GridWorld(config, "paper_map")
        policy = {state: "down" for state in world.nonterminal}
        with tempfile.TemporaryDirectory() as folder:
            episodes, _ = simulate(world, policy, Path(folder))
        self.assertEqual((episodes[0]["steps"], episodes[0]["outcome"]), (100, "time limit"))
        self.assertAlmostEqual(episodes[0]["discounted_return"], -10 * (1 - 0.9 ** 100))


if __name__ == "__main__":
    unittest.main()
