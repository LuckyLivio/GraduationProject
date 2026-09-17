import unittest
from unittest.mock import patch

import numpy as np

from foresight import comparison
from foresight.env import NavigationEnv, damping_at, physics_step
from foresight.planner import PhysicsPredictor


class ConstantLearnedModel:
    """Tiny stand-in; these tests validate comparison contracts, not accuracy."""
    n_members = 3

    def __init__(self):
        self.observed = []

    def damping(self, positions):
        return np.ones(np.asarray(positions).shape[:-1], dtype=np.float32)

    def observe(self, state, action, next_state):
        self.observed.append(np.asarray(state).copy())

    def rollout(self, state, actions):
        return np.repeat(PhysicsPredictor(1).rollout(state, actions), 3, axis=0)

    def reset(self):
        self.observed.clear()


class ComparisonTests(unittest.TestCase):
    def test_common_actions_and_metrics_use_physical_branch(self):
        base = ConstantLearnedModel()
        result = comparison.build_comparison(base, .1, .8)
        self.assertEqual([m['id'] for m in result['methods']], list(comparison.METHOD_IDS))
        self.assertEqual(len(result['history']), 9)
        self.assertEqual(result['executed_steps'], 16)
        state = np.asarray(result['snapshot'], dtype=np.float32)
        actual = [state.copy()]
        for action in result['actions']:
            state = physics_step(state, action, damping_at(state[:2]))
            actual.append(state.copy())
        np.testing.assert_array_equal(result['actual_path'], actual)
        for method in result['methods']:
            errors = np.linalg.norm(np.asarray(method['predicted_path'])[:, :2] - np.asarray(actual)[:, :2], axis=-1)
            self.assertAlmostEqual(method['position_rmse'], float(np.sqrt(np.mean(errors[1:] ** 2))))
            self.assertAlmostEqual(method['endpoint_error'], float(errors[-1]))
        self.assertEqual(base.observed, [], 'comparison must not mutate caller model')
        self.assertEqual(result['methods'][1]['updates'], 8)

    def test_predictions_precede_future_truth_and_have_no_future_updates(self):
        events = []
        original_step = NavigationEnv.step
        original_rollout = ConstantLearnedModel.rollout

        def step(env, action):
            events.append(('truth', env.steps))
            return original_step(env, action)

        def rollout(model, state, actions):
            events.append(('prediction', len(model.observed)))
            return original_rollout(model, state, actions)

        with patch.object(NavigationEnv, 'step', step), patch.object(ConstantLearnedModel, 'rollout', rollout):
            comparison.build_comparison(ConstantLearnedModel(), .1, .8)
        prediction_index = next(i for i, item in enumerate(events) if item[0] == 'prediction')
        future_index = events.index(('truth', 8))
        self.assertLess(prediction_index, future_index)
        self.assertEqual(events[prediction_index][1], 8)

    def test_early_terminal_scores_only_executed_prefix(self):
        class EarlyEnv(NavigationEnv):
            def step(self, action):
                state, reward, terminated, truncated, info = super().step(action)
                if self.steps == 10:
                    self.done, truncated, info['timeout'] = True, True, True
                return state, reward, terminated, truncated, info
        with patch.object(comparison, 'NavigationEnv', EarlyEnv):
            result = comparison.build_comparison(ConstantLearnedModel(), .1, .8)
        self.assertEqual(result['executed_steps'], 2)
        self.assertTrue(result['terminal']['timeout'])
        for method in result['methods']:
            self.assertEqual(len(method['predicted_path']), 17)
            self.assertEqual(len(method['position_errors']), 3)
            self.assertEqual(method['evaluated_steps'], 2)

    def test_action_presets_are_declared_and_model_independent(self):
        result = comparison.build_comparison(ConstantLearnedModel(), .1, .8,
                                             condition='global_shift', action_mode='brake')
        np.testing.assert_allclose(np.asarray(result['actions'])[:4], np.tile([-.2, 0], (4, 1)))
        np.testing.assert_array_equal(np.asarray(result['actions'])[4:], 0)
        with self.assertRaises(ValueError):
            comparison.build_comparison(ConstantLearnedModel(), .1, .8, horizon=33)
        with self.assertRaises(ValueError):
            comparison.build_comparison(ConstantLearnedModel(), .1, .8, condition='unknown')

    def test_navigation_is_paired_and_follows_episode_termination(self):
        class ShortEnv(NavigationEnv):
            def step(self, action):
                state, reward, terminated, truncated, info = super().step(action)
                if self.steps >= 2:
                    self.done, truncated, info['timeout'] = True, True, True
                return state, reward, terminated, truncated, info
        with patch.object(comparison, 'NavigationEnv', ShortEnv):
            result = comparison.build_navigation(ConstantLearnedModel(), .1, .8)
        for method in result['methods']:
            self.assertEqual(method['steps'], 2)
            self.assertTrue(method['timeout'])
            np.testing.assert_array_equal(method['frames'][0]['state'], result['initial_state'])
            np.testing.assert_array_equal(method['frames'][-1]['next_state'], method['actual_path'][-1])
            self.assertLessEqual(method['frames'][0]['model_steps'], 4500)
            self.assertGreater(method['mean_controller_ms'], 0)


if __name__ == '__main__':
    unittest.main()
