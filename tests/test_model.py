"""World-model contracts independent of stochastic training convergence."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from foresight.env import NavigationEnv, physics_step
from foresight.model import HybridWorldModel, transition_targets


def constant_damping_member(damping: float):
    """Known weights whose sigmoid output is exactly the requested damping."""
    probability = (damping - 0.02) / 5.98
    logit = np.log(probability / (1.0 - probability))
    return [np.zeros((2, 4), dtype=np.float32), np.zeros(4, dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32), np.zeros(4, dtype=np.float32),
            np.zeros((4, 1), dtype=np.float32), np.asarray([logit], dtype=np.float32)]


class WorldModelTests(unittest.TestCase):
    def test_inverse_recovers_unsaturated_damping_and_filters_nonidentifiable_rows(self):
        base = NavigationEnv(seed=4).state
        base[:2] = [5.0, 5.0]
        states = np.broadcast_to(base, (5, 18)).copy()
        states[:, 2:4] = [[0.7, -0.3], [-0.5, 0.4], [0, 0], [1.79, 0], [-1.0, 0]]
        states[-1, :2] = [0.19, 5.0]
        actions = np.asarray([[0.2, -0.1], [-0.3, 0.2], [0.2, 0], [1.0, 0], [-1.0, 0]], dtype=np.float32)
        actual_damping = np.asarray([0.6, 2.1, 0.8, 0.1, 0.8], dtype=np.float32)
        next_states = physics_step(states, actions, actual_damping)
        positions, inferred = transition_targets({"states": states, "actions": actions,
                                                   "next_states": next_states})
        np.testing.assert_array_equal(positions, states[:2, :2])
        np.testing.assert_allclose(inferred, actual_damping[:2], rtol=1e-5, atol=1e-5)
        self.assertEqual(len(inferred), 2, "Low-speed, clipped-speed and wall-clamped transitions must be excluded")

    def test_action_conditional_rollout_matches_known_hybrid_dynamics(self):
        damping_values = [0.6, 2.1]
        model = HybridWorldModel([constant_damping_member(d) for d in damping_values])
        state = NavigationEnv(seed=1).state
        state[:4] = [5.0, 5.0, 0.4, 0.0]
        actions = np.zeros((2, 5, 2), dtype=np.float32)
        actions[0, :, 0] = 0.4
        actions[1, :, 0] = -0.4
        predicted = model.rollout(state, actions)
        self.assertEqual(predicted.shape, (2, 2, 6, 18))
        for member, damping in enumerate(damping_values):
            for candidate in range(2):
                truth = state.copy()
                np.testing.assert_array_equal(predicted[member, candidate, 0], truth)
                for step in range(5):
                    truth = physics_step(truth, actions[candidate, step], damping)
                    np.testing.assert_allclose(predicted[member, candidate, step + 1], truth, rtol=1e-5, atol=1e-6)
        self.assertGreater(predicted[0, 0, -1, 0], predicted[0, 1, -1, 0])
        self.assertGreater(predicted[0, 0, -1, 0], predicted[1, 0, -1, 0])
        # Frozen offline model must not adapt itself from evaluation episodes.
        model.observe(state, actions[0, 0], predicted[0, 0, 1])
        model.reset()
        np.testing.assert_array_equal(model.rollout(state, actions), predicted)

    def test_save_load_preserves_all_predictions_without_pickle(self):
        model = HybridWorldModel([constant_damping_member(0.9), constant_damping_member(1.7)])
        state = NavigationEnv(seed=6).state
        actions = np.random.default_rng(71).uniform(-0.5, 0.5, (3, 4, 2)).astype(np.float32)
        expected = model.rollout(state, actions)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            model.save(path)
            with np.load(path, allow_pickle=False) as stored:
                self.assertEqual(len(stored.files), 12)
                self.assertTrue(all(stored[name].dtype != object for name in stored.files))
            restored = HybridWorldModel.load(path)
        self.assertEqual(restored.n_members, 2)
        np.testing.assert_array_equal(restored.rollout(state, actions), expected)


if __name__ == "__main__":
    unittest.main()
