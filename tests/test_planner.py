import unittest

import numpy as np

from foresight.planner import MPCPlanner, PhysicsPredictor


def sample_state():
    return np.array([1, 1, 0.3, 0.1, 9, 9, 3, 7, 0.2, 0, 7, 3, 0, 0.2, 6, 6, -0.2, 0.1], dtype=np.float32)


class CountingEnsemble:
    n_members = 3

    def __init__(self):
        self.steps = 0
        self.models = [PhysicsPredictor(damping=d) for d in (0.4, 1.0, 2.0)]

    def rollout(self, state, actions):
        self.steps += self.n_members * actions.shape[0] * actions.shape[1]
        return np.concatenate([m.rollout(state, actions) for m in self.models], axis=0)


class PlannerTests(unittest.TestCase):
    def test_action_conditioned_rollout(self):
        actions = np.zeros((2, 10, 2), dtype=np.float32)
        actions[1, :, 0] = 1
        paths = PhysicsPredictor().rollout(sample_state(), actions)
        self.assertEqual(paths.shape, (1, 2, 11, 18))
        self.assertGreater(paths[0, 1, -1, 0], paths[0, 0, -1, 0] + 0.5)

    def test_budget_includes_adaptive_probe(self):
        for adaptive in (False, True):
            predictor = CountingEnsemble()
            planner = MPCPlanner(predictor, adaptive=adaptive, budget=1500)
            action, info = planner.act(sample_state())
            self.assertEqual(info["model_steps"], predictor.steps)
            self.assertLessEqual(info["model_steps"], 1500)
            self.assertTrue(np.all(np.abs(action) <= 1))
            self.assertTrue(np.all(np.isfinite(action)))
            self.assertEqual(len(info["predicted_path"]), info["horizon"] + 1)
            if adaptive:
                self.assertEqual(len(info["probe_disagreement"]), 17)

    def test_deterministic_first_action(self):
        a, ia = MPCPlanner(PhysicsPredictor(), budget=2000, seed=10).act(sample_state())
        b, ib = MPCPlanner(PhysicsPredictor(), budget=2000, seed=10).act(sample_state())
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ia["predicted_path"], ib["predicted_path"])

    def test_uncertainty_changes_horizon(self):
        _, short = MPCPlanner(CountingEnsemble(), adaptive=True, budget=1500, uncertainty_threshold=0).act(sample_state())
        _, long = MPCPlanner(CountingEnsemble(), adaptive=True, budget=1500, uncertainty_threshold=100).act(sample_state())
        self.assertEqual(short["horizon"], 5)
        self.assertEqual(long["horizon"], 16)

    def test_local_identifier_uses_only_history(self):
        predictor = PhysicsPredictor(adaptive=True)
        truth = PhysicsPredictor(damping=1.7)
        state = sample_state()
        action = np.array([0.2, 0.3], dtype=np.float32)
        next_state = truth.rollout(state, action[None, None])[0, 0, 1]
        predictor.observe(state, action, next_state)
        self.assertAlmostEqual(predictor.damping, 1.7, places=5)
        predictor.reset()
        self.assertEqual(predictor.damping, 0.8)

    def test_rejects_unaffordable_probe(self):
        with self.assertRaises(ValueError):
            MPCPlanner(CountingEnsemble(), adaptive=True, budget=10)

    def test_swept_collision_catches_obstacle_between_endpoints(self):
        state = sample_state()
        state[:2] = (4, 4)
        state[6:10] = (5, 4, 0, 0)
        paths = np.broadcast_to(state, (1, 1, 2, 18)).copy()
        paths[0, 0, 1, :2] = (6, 4)
        _, collision = MPCPlanner._cost(paths, np.zeros((1, 1, 2)))
        self.assertTrue(collision[0])

    def test_returned_info_is_json_ready(self):
        import json
        _, info = MPCPlanner(PhysicsPredictor(), budget=500).act(sample_state())
        encoded = json.dumps(info, allow_nan=False)
        self.assertIn('predicted_path', encoded)


if __name__ == "__main__":
    unittest.main()
