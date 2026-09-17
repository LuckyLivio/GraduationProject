import unittest

import numpy as np

from foresight.env import physics_step
from foresight.model import HybridWorldModel
from foresight.planner import MPCPlanner
from foresight.residual import ResidualCalibratedModel


def constant_model(damping=1.2):
    fraction = (damping - .02) / 5.98
    bias = np.array([np.log(fraction / (1 - fraction))], dtype=np.float32)
    return HybridWorldModel([
        [np.zeros((2, 32), dtype=np.float32), np.zeros(32, dtype=np.float32),
         np.zeros((32, 32), dtype=np.float32), np.zeros(32, dtype=np.float32),
         np.zeros((32, 1), dtype=np.float32), bias.copy()]
        for _ in range(3)
    ])


def moving_state():
    return np.array([2, 2, .7, -.3, 9, 7, 3, 8, .1, 0, 6, 8, 0, .2, 8, 3, -.1, 0], dtype=np.float32)


class ResidualTests(unittest.TestCase):
    def test_recovers_shift_from_observed_transition(self):
        model = ResidualCalibratedModel(constant_model())
        state = moving_state()
        action = np.array([.2, -.1], dtype=np.float32)
        ns = physics_step(state, action, 1.2 * 1.7)
        model.observe(state, action, ns)
        self.assertAlmostEqual(model.current_scale, 1.7, places=5)
        self.assertEqual(model.updates, 1)
        self.assertEqual(model.calibration_model_evaluations, 3)
        predicted = model.rollout(state, action[None, None])
        np.testing.assert_allclose(predicted[:, 0, 1], np.broadcast_to(ns, (3, 18)), atol=1e-6)

    def test_base_weights_and_outputs_are_unchanged(self):
        base = constant_model()
        before = [[weight.copy() for weight in member] for member in base.members]
        state = moving_state()
        actions = np.zeros((1, 10, 2), dtype=np.float32)
        base_prediction = base.rollout(state, actions)
        model = ResidualCalibratedModel(base)
        model.observe(state, actions[0, 0], physics_step(state, actions[0, 0], 2.0))
        model.rollout(state, actions)
        model.reset()
        for member, original in zip(base.members, before):
            for weight, old in zip(member, original):
                np.testing.assert_array_equal(weight, old)
        np.testing.assert_array_equal(base.rollout(state, actions), base_prediction)

    def test_reset_clears_history_scale_and_accounting(self):
        model = ResidualCalibratedModel(constant_model())
        state, action = moving_state(), np.zeros(2)
        model.observe(state, action, physics_step(state, action, 2.0))
        model.reset()
        self.assertEqual(model.current_scale, 1.0)
        self.assertEqual(len(model.history), 0)
        self.assertEqual(model.updates, 0)
        self.assertEqual(model.calibration_model_evaluations, 0)
        self.assertEqual(model.calibration_ms, 0)

    def test_scale_clipping(self):
        state, action = moving_state(), np.zeros(2)
        for damping, expected in ((.1, .5), (5.0, 2.5)):
            model = ResidualCalibratedModel(constant_model())
            model.observe(state, action, physics_step(state, action, damping))
            self.assertEqual(model.current_scale, expected)

    def test_invalid_transitions_do_not_update(self):
        model = ResidualCalibratedModel(constant_model())
        state, action = moving_state(), np.zeros(2)
        stopped = state.copy()
        stopped[2:4] = 0
        model.observe(stopped, action, physics_step(stopped, action, 2.0))
        saturated = state.copy()
        saturated[2:4] = (1.8, 0)
        model.observe(saturated, np.ones(2), physics_step(saturated, np.ones(2), .1))
        wall = state.copy()
        wall[:2] = (.181, 2)
        wall[2:4] = (-.8, 0)
        model.observe(wall, action, physics_step(wall, action, 1.0))
        self.assertEqual(model.updates, 0)
        self.assertEqual(model.calibration_model_evaluations, 0)
        self.assertEqual(model.current_scale, 1.0)

    def test_weighted_fit_and_rolling_window(self):
        model = ResidualCalibratedModel(constant_model(1.0), window=2)
        action = np.zeros(2)
        for speed, damping in ((.4, .6), (.8, 1.4), (.5, 2.0)):
            state = moving_state()
            state[2:4] = (speed, 0)
            model.observe(state, action, physics_step(state, action, damping))
        expected = (.8 ** 2 * 1.4 + .5 ** 2 * 2.0) / (.8 ** 2 + .5 ** 2)
        self.assertAlmostEqual(model.current_scale, expected, places=5)
        self.assertEqual(len(model.history), 2)
        self.assertEqual(model.calibration_model_evaluations, 9)

    def test_fixed_horizon_planner_contract_and_accounting(self):
        model = ResidualCalibratedModel(constant_model())
        planner = MPCPlanner(model, horizon=10, budget=1500, seed=0)
        state = moving_state()
        action, info = planner.act(state)
        self.assertEqual(info["horizon"], 10)
        self.assertEqual(info["model_steps"], 1500)
        self.assertEqual(model.calibration_model_evaluations, 0)
        planner.observe(state, action, physics_step(state, action, 2.0))
        self.assertEqual(model.calibration_model_evaluations, 3)
        self.assertGreaterEqual(model.calibration_ms, 0)


if __name__ == "__main__":
    unittest.main()
