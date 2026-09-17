"""Tests cover physical/reproducibility contracts used by every experiment."""
import unittest

import numpy as np

from foresight.data import collect_dataset
from foresight.env import DT, EnvConfig, NavigationEnv, damping_at, physics_step


class EnvironmentTests(unittest.TestCase):
    def test_seeded_episode_and_returned_state_are_independent(self):
        a, b = NavigationEnv(seed=52), NavigationEnv(seed=52)
        np.testing.assert_array_equal(a.state, b.state)
        for _ in range(8):
            np.testing.assert_array_equal(a.step([0.5, -0.1])[0], b.step([0.5, -0.1])[0])
        escaped = a.state
        escaped[:] = -999
        self.assertTrue(np.all(a.state > -999))
        trace = a.trajectory
        trace[0][:] = -999
        self.assertTrue(np.all(a.trajectory[0] > -999))

    def test_vectorized_dynamics_action_and_hidden_damping(self):
        env = NavigationEnv(seed=1)
        state = env.state
        state[2:4] = [1.0, 0.0]
        states = np.broadcast_to(state, (2, 3, 18))
        actions = np.zeros((2, 3, 2), dtype=np.float32)
        actions[1, :, 1] = 1.0
        outputs = physics_step(states, actions, np.array([[0.5], [2.0]]))
        self.assertEqual(outputs.shape, (2, 3, 18))
        self.assertGreater(outputs[0, 0, 2], outputs[1, 0, 2])
        self.assertGreater(outputs[1, 0, 3], outputs[0, 0, 3])
        for i in range(2):
            for j in range(3):
                np.testing.assert_allclose(outputs[i, j], physics_step(state, actions[i, j], [0.5, 2.0][i]))

    def test_exact_reflection_for_long_obstacle_travel(self):
        state = NavigationEnv(seed=3).state
        state[6:10] = [9.5, 5.0, 4.0, 0.0]
        result = physics_step(state, [0, 0], 1.0)
        self.assertAlmostEqual(float(result[6]), 9.14, places=5)
        self.assertAlmostEqual(float(result[8]), -4.0, places=5)

    def test_swept_collision_does_not_tunnel(self):
        env = NavigationEnv(seed=4)
        state = env.state
        state[:4] = [5.0, 5.0, 0.0, 0.0]
        state[6:10] = [4.0, 5.0, 2.0 / DT, 0.0]
        state[10:14] = [2.0, 2.0, 0.0, 0.0]
        state[14:18] = [8.0, 8.0, 0.0, 0.0]
        env.state = state
        _, _, terminated, truncated, info = env.step([0.0, 0.0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["collision"])
        self.assertLess(info["min_clearance"], 0)

    def test_collision_detected_on_reflected_piece(self):
        env = NavigationEnv(seed=4)
        state = env.state
        state[:4] = [8.8, 5.0, 0.0, 0.0]
        state[6:10] = [9.5, 5.0, 20.0, 0.0]
        state[10:14] = [2.0, 2.0, 0.0, 0.0]
        state[14:18] = [3.0, 8.0, 0.0, 0.0]
        env.state = state
        _, _, terminated, _, info = env.step([0.0, 0.0])
        self.assertTrue(terminated)
        self.assertTrue(info["collision"])

    def test_ood_changes_transition_without_changing_observation(self):
        a = NavigationEnv(seed=8, damping_scale=1)
        b = NavigationEnv(seed=8, damping_scale=2)
        np.testing.assert_array_equal(a.state, b.state)
        state = a.state
        state[2:4] = [0.7, 0.2]
        a.state = state
        b.state = state
        next_a = a.step([0.1, 0.0])[0]
        next_b = b.step([0.1, 0.0])[0]
        self.assertGreater(np.linalg.norm(next_a[2:4]), np.linalg.norm(next_b[2:4]))
        self.assertEqual(next_a.shape, (18,))
        self.assertGreater(float(damping_at([9, 4])), float(damping_at([1, 4])))

    def test_timeout_and_reset(self):
        env = NavigationEnv(seed=1, scenario="open", config=EnvConfig(max_steps=2))
        self.assertFalse(env.step([0, 0])[3])
        _, _, terminated, truncated, info = env.step([0, 0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["timeout"])
        with self.assertRaises(RuntimeError):
            env.step([0, 0])
        env.reset(seed=1)
        self.assertEqual(env.steps, 0)

    def test_wall_collision_and_goal_success_are_terminal(self):
        wall = NavigationEnv(seed=1, scenario="open")
        state = wall.state
        state[:4] = [0.20, 5.0, -1.0, 0.0]
        wall.state = state
        _, _, terminated, truncated, info = wall.step([-1, 0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["collision"])
        goal = NavigationEnv(seed=1, scenario="open")
        state = goal.state
        state[4:6] = state[:2] + [0.3, 0.0]
        goal.state = state
        _, reward, terminated, truncated, info = goal.step([0, 0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertGreater(reward, 9)

    def test_dataset_episode_partition_and_reproducibility(self):
        data_a = collect_dataset(episodes=3, seed=12, max_steps=6)
        data_b = collect_dataset(episodes=3, seed=12, max_steps=6)
        self.assertEqual(data_a["states"].shape, (18, 18))
        np.testing.assert_array_equal(np.unique(data_a["episode_ids"]), [0, 1, 2])
        for key in data_a:
            np.testing.assert_array_equal(data_a[key], data_b[key])


if __name__ == "__main__":
    unittest.main()
