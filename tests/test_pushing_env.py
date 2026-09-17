"""Behavioral checks for contact physics and observation boundaries."""

import json
import unittest

import numpy as np

from foresight.pushing_env import PushParams, generate_dataset, simulate_push


class PushingEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.state = np.zeros(6)
        self.params = PushParams(damping=0.3, friction=0.6)

    def test_centered_push_translates_through_contact(self):
        result, info = simulate_push(self.state, np.array([0, 0, 0.6]), self.params)
        self.assertTrue(info["contact"])
        self.assertGreater(info["contact_steps"], 0)
        self.assertGreater(info["total_impulse"], 0)
        self.assertGreater(result[0], 0.1)
        self.assertLess(abs(result[1]), 1e-6)
        self.assertLess(abs(result[2]), 1e-6)

    def test_offset_contact_changes_rotation_sign(self):
        upper, _ = simulate_push(self.state, np.array([0, 0.65, 0.6]), self.params)
        lower, _ = simulate_push(self.state, np.array([0, -0.65, 0.6]), self.params)
        self.assertLess(upper[2], -0.02)
        self.assertGreater(lower[2], 0.02)
        self.assertAlmostEqual(upper[2], -lower[2], places=6)

    def test_reproducible_without_mutating_inputs(self):
        state = np.array([0.1, -0.2, 0.5, 0.01, -0.01, 0.02])
        action = np.array([2.0, -0.3, 0.4])
        before_state, before_action = state.copy(), action.copy()
        params = PushParams(0.15, 0.7, 0.025, -0.035)
        first, first_info = simulate_push(state, action, params, record=True)
        second, second_info = simulate_push(state, action, params, record=True)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(state, before_state)
        np.testing.assert_array_equal(action, before_action)
        self.assertEqual(first_info, second_info)
        np.testing.assert_allclose(first_info["frames"][0], state, atol=1e-12)

    def test_observation_and_info_do_not_reveal_hidden_parameters(self):
        result, info = simulate_push(self.state, np.array([1, 0.2, 0.4]), self.params, True)
        self.assertEqual(result.shape, (6,))
        self.assertEqual(set(info), {"contact", "contact_steps", "total_impulse", "duration",
                                     "frames", "frame_times", "pusher_positions"})
        self.assertEqual(len(info["frames"]), 11)
        self.assertEqual(len(info["pusher_positions"]), 11)
        self.assertIsNone(info["pusher_positions"][-1])
        np.testing.assert_array_equal(info["frames"][-1], result)
        json.dumps(info, allow_nan=False)

    def test_hidden_dynamics_changes_the_same_action_outcome(self):
        action = np.array([0, 0, 0.6])
        light_damping, _ = simulate_push(self.state, action, PushParams(0.8, 0.6))
        heavy_damping, _ = simulate_push(self.state, action, PushParams(0.05, 0.6))
        self.assertGreater(light_damping[0], heavy_damping[0] + 0.015)
        offset_cog, _ = simulate_push(self.state, action, PushParams(0.3, 0.6, 0, 0.05))
        self.assertGreater(abs(offset_cog[2]), 0.02)

    def test_complete_episode_collection_is_deterministic(self):
        first = generate_dataset(2, 3, seed=12)
        second = generate_dataset(2, 3, seed=12)
        self.assertEqual(first["states"].shape, (2, 4, 6))
        self.assertEqual(first["actions"].shape, (2, 3, 3))
        self.assertEqual(first["contacts"].shape, (2, 3))
        self.assertEqual(first["hidden_params"].shape, (2, 4))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_rejects_invalid_physics_or_actions(self):
        with self.assertRaises(ValueError):
            PushParams(0.0, 0.5)
        with self.assertRaises(ValueError):
            PushParams(0.5, 0.5, cog_x=0.08)
        for action in ([0.5, 0.0, 0.5], [0, 1, 0.5], [0, 0, 2]):
            with self.assertRaises(ValueError):
                simulate_push(self.state, np.array(action), self.params)


if __name__ == "__main__":
    unittest.main()
