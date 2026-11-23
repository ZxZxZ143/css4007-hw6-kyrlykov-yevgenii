"""Check the episode runner's terminal/time-limit boundary and real route."""

import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from part_b import run_episode


class LastStepEnvironment:
    """Hold at start until step 100, then terminate and/or truncate."""

    def __init__(self, terminal_tile=None):
        self.tile = terminal_tile
        self.unwrapped = SimpleNamespace(desc=np.array([
            list("SFFF"), list("FHFH"), list("FFFH"), list("HFFG")
        ], dtype="S1"))

    def reset(self, *, seed):
        self.steps = 0
        return 0, {}

    def step(self, action):
        self.steps += 1
        last = self.steps == 100
        state = (15 if self.tile == "G" else 5) if last and self.tile else 0
        return state, float(last and self.tile == "G"), bool(last and self.tile), last, {}

    def render(self):
        return "SFFF\nFHFH\nFFFH\nHFFG\n"


class EpisodeTests(unittest.TestCase):
    def test_goal_on_step_100_wins_over_truncation(self):
        result, _ = run_episode(LastStepEnvironment("G"), "fixed", 11)
        self.assertEqual((result["outcome"], result["steps"], result["total_reward"]),
                         ("goal", 100, 1.0))

    def test_hole_on_step_100_wins_over_truncation(self):
        result, _ = run_episode(LastStepEnvironment("H"), "fixed", 11)
        self.assertEqual((result["outcome"], result["steps"]), ("hole", 100))

    def test_no_terminal_state_is_time_limit(self):
        result, _ = run_episode(LastStepEnvironment(), "fixed", 11)
        self.assertEqual((result["outcome"], result["steps"]), ("time limit", 100))

    def test_real_fixed_route_without_slipping(self):
        env = gym.make("FrozenLake-v1", map_name="4x4",
                       is_slippery=False, render_mode="ansi")
        try:
            result, trace = run_episode(env, "fixed", 11000)
            self.assertEqual((result["outcome"], result["steps"]), ("goal", 6))
            for state in (4, 8, 9, 13, 14, 15):
                self.assertIn(f"next_state={state},", trace)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
