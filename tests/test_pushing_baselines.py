import unittest

import numpy as np

from foresight.pushing_baselines import (
    goal_cost,
    identify_parameters,
    parameter_candidates,
    predict_physics,
)
from foresight.pushing_env import PushParams, sample_action


class PushingBaselineTests(unittest.TestCase):
    def test_history_identification_predicts_unseen_pushes(self):
        """A fitted baseline must generalize beyond its fitting transitions."""
        true_params = PushParams(.12, .82, .055, -.04)
        wrong_params = PushParams(.75, .15, -.06, .065)
        start = np.array([0., 0., .2, 0., 0., 0.])
        history_actions = np.array([[0, -.65, .8], [1, .6, .7], [2, .35, .9], [3, -.5, .6]])
        history_states = predict_physics(start, history_actions, true_params)
        states_before = history_states.copy()
        actions_before = history_actions.copy()
        fitted, info = identify_parameters(history_states, history_actions, [wrong_params, true_params])
        test_actions = np.array([[2, -.7, .75], [0, .55, .65]])
        expected = predict_physics(history_states[-1], test_actions, true_params)
        actual = predict_physics(history_states[-1], test_actions, fitted)
        wrong = predict_physics(history_states[-1], test_actions, wrong_params)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        self.assertGreater(float(np.linalg.norm(wrong - expected)), 1e-5)
        self.assertEqual(info["history_steps"], 4)
        self.assertEqual(info["model_calls"], 8)
        self.assertLess(info["fit_rmse"], 1e-10)
        np.testing.assert_array_equal(history_states, states_before)
        np.testing.assert_array_equal(history_actions, actions_before)

    def test_no_history_is_explicitly_not_an_identification(self):
        estimated, info = identify_parameters(np.zeros((1, 6)), np.empty((0, 3)))
        self.assertEqual(info["status"], "no_history")
        self.assertIsNone(info["fit_rmse"])
        self.assertEqual(info["model_calls"], 0)
        self.assertAlmostEqual(estimated.damping, .5)
        self.assertAlmostEqual(estimated.friction, .55)

    def test_default_fit_handles_damping_outside_training_range(self):
        rng = np.random.default_rng(932)
        true_params = PushParams(.9, .62, .047, -.034)
        history_actions = np.array([sample_action(rng) for _ in range(6)])
        history_states = predict_physics(np.zeros(6), history_actions, true_params)
        fitted, info = identify_parameters(history_states, history_actions)
        future_actions = np.array([sample_action(rng) for _ in range(3)])
        truth = predict_physics(history_states[-1], future_actions, true_params)
        estimated = predict_physics(history_states[-1], future_actions, fitted)
        nominal = predict_physics(history_states[-1], future_actions, PushParams(.5, .55))
        fitted_error = np.linalg.norm(estimated[1:, :2] - truth[1:, :2])
        nominal_error = np.linalg.norm(nominal[1:, :2] - truth[1:, :2])
        self.assertLess(fitted_error, .7 * nominal_error)
        self.assertLessEqual(info["fit_rmse"], info["initial_best_rmse"])
        self.assertEqual(info["model_calls"], 6 * info["hypotheses_evaluated"])

    def test_default_bank_is_reproducible_and_does_not_consume_global_rng(self):
        np.random.seed(134)
        expected = np.random.random(3)
        np.random.seed(134)
        first = parameter_candidates()
        np.testing.assert_array_equal(np.random.random(3), expected)
        second = parameter_candidates()
        self.assertEqual(len(first), 97)
        for left, right in zip(first, second):
            self.assertEqual(left, right)

    def test_angle_wrap_does_not_create_a_false_large_pose_error(self):
        states = np.zeros((2, 2, 6))
        states[0, -1, 2] = -np.pi + .01
        states[1, -1, 2] = np.pi - .01
        goal = np.array([0., 0., np.pi - .01])
        costs = goal_cost(states, goal)
        np.testing.assert_allclose(costs, [(.25 * .02) ** 2, 0.], atol=1e-12)
        self.assertIsInstance(goal_cost(states[0], goal), float)

    def test_empty_rollout_retains_start_without_mutating_it(self):
        state = np.array([1., -.4, .5, .02, 0., -.01])
        before = state.copy()
        prediction = predict_physics(state, [], PushParams(.4, .5))
        np.testing.assert_array_equal(prediction, state[None])
        prediction[0, 0] = 999
        np.testing.assert_array_equal(state, before)

    def test_inconsistent_or_nonfinite_history_is_rejected(self):
        cases = [
            (np.zeros((2, 6)), np.empty((0, 3))),
            (np.zeros((2, 5)), np.zeros((1, 3))),
            (np.full((2, 6), np.nan), np.zeros((1, 3))),
        ]
        for states, actions in cases:
            with self.subTest(shape=states.shape):
                with self.assertRaises(ValueError):
                    identify_parameters(states, actions)
        with self.assertRaises(ValueError):
            identify_parameters(np.zeros((1, 6)), [], [])


if __name__ == "__main__":
    unittest.main()
