import unittest

import numpy as np

from foresight.env import physics_step
from foresight.gated import (GatedCalibratedModel, MatchedHistoryPhysicsModel,
                             gate_threshold_from_validation)
from foresight.model import HybridWorldModel
from foresight.residual import ResidualCalibratedModel


def constant_model(damping=1.0):
    fraction = (damping - .02) / 5.98
    bias = np.array([np.log(fraction / (1 - fraction))], dtype=np.float32)
    return HybridWorldModel([
        [np.zeros((2, 32), dtype=np.float32), np.zeros(32, dtype=np.float32),
         np.zeros((32, 32), dtype=np.float32), np.zeros(32, dtype=np.float32),
         np.zeros((32, 1), dtype=np.float32), bias.copy()]
        for _ in range(3)
    ])


def moving_state(speed=.7):
    return np.array([2, 2, speed, -.3, 9, 7, 3, 8, .1, 0, 6, 8, 0, .2, 8, 3, -.1, 0], dtype=np.float32)


def observation(model, damping, speed=.7):
    state = moving_state(speed)
    action = np.zeros(2, dtype=np.float32)
    model.observe(state, action, physics_step(state, action, damping))


class GatedTests(unittest.TestCase):
    def test_inactive_rollout_is_exactly_frozen_despite_small_fit_error(self):
        base = constant_model()
        model = GatedCalibratedModel(base, threshold=.15)
        actions = np.ones((3, 12, 2), dtype=np.float32) * .1
        for _ in range(15):
            observation(model, 1.04)
        self.assertFalse(model.gate_active)
        self.assertAlmostEqual(model.raw_scale, 1.04, places=5)
        self.assertEqual(model.current_scale, 1.0)
        np.testing.assert_array_equal(model.rollout(moving_state(), actions),
                                      base.rollout(moving_state(), actions))
        self.assertEqual(model.calibration_model_evaluations, 15 * 3)

    def test_minimum_history_and_persistence_delay_activation(self):
        model = GatedCalibratedModel(constant_model(), threshold=.15)
        for count in range(1, 6):
            observation(model, 1.7)
            self.assertEqual(model.updates, count)
            self.assertEqual(model.gate_active, count == 5)
        self.assertAlmostEqual(model.current_scale, 1.7, places=5)
        self.assertEqual(model.gate_on_events, 1)
        self.assertEqual(model.gate_off_events, 0)

    def test_recovery_requires_window_flush_and_consecutive_low_scores(self):
        model = GatedCalibratedModel(constant_model(), threshold=.15)
        for _ in range(5):
            observation(model, 1.7)
        for _ in range(6):
            observation(model, 1.0)
            self.assertTrue(model.gate_active)
        observation(model, 1.0)
        self.assertFalse(model.gate_active)
        self.assertEqual(model.current_scale, 1)
        self.assertEqual(model.gate_off_events, 1)

    def test_middle_band_breaks_consecutive_streaks(self):
        model = GatedCalibratedModel(constant_model(), .2, window=1,
                                    min_samples=1, persistence=3, off_ratio=.5)
        for _ in range(2):
            observation(model, 1.7)
        observation(model, np.exp(.15))
        self.assertEqual(model.on_streak, 0)
        for _ in range(2):
            observation(model, 1.7)
        self.assertFalse(model.gate_active)
        observation(model, 1.7)
        self.assertTrue(model.gate_active)
        for _ in range(2):
            observation(model, 1.0)
        observation(model, np.exp(.15))
        self.assertEqual(model.off_streak, 0)
        for _ in range(2):
            observation(model, 1.0)
        self.assertTrue(model.gate_active)
        observation(model, 1.0)
        self.assertFalse(model.gate_active)

    def test_invalid_transitions_keep_gate_history_and_streaks(self):
        model = GatedCalibratedModel(constant_model(), threshold=.15)
        for _ in range(4):
            observation(model, 1.7)
        stopped, action = moving_state(), np.zeros(2)
        stopped[2:4] = 0
        for _ in range(4):
            model.observe(stopped, action, physics_step(stopped, action, 1.0))
        self.assertFalse(model.gate_active)
        self.assertEqual(model.on_streak, 2)
        self.assertEqual(model.updates, 4)
        self.assertEqual(len(model.history), 4)
        self.assertEqual(model.last_calibration_model_evaluations, 0)
        observation(model, 1.7)
        self.assertTrue(model.gate_active)
        self.assertEqual(model.observations, 9)

    def test_raw_fit_matches_ungated_fit_and_rollover(self):
        base = constant_model()
        gated = GatedCalibratedModel(base, .15, window=3)
        ungated = ResidualCalibratedModel(base, window=3)
        for speed, damping in ((.5, 1.0), (.8, 1.8), (.7, .6), (.4, 2.0), (.9, 1.5)):
            observation(gated, damping, speed)
            observation(ungated, damping, speed)
            self.assertEqual(gated.raw_scale, ungated.current_scale)
        self.assertEqual(len(gated.history), 3)
        self.assertEqual(gated.calibration_model_evaluations, ungated.calibration_model_evaluations)

    def test_reset_clears_gate_and_preserves_base_weights(self):
        base = constant_model()
        before = [[w.copy() for w in member] for member in base.members]
        model = GatedCalibratedModel(base, threshold=.15)
        for _ in range(7):
            observation(model, 1.7)
        model.reset()
        self.assertFalse(model.gate_active)
        for attr in ("gate_on_events", "gate_off_events", "gate_score", "on_streak",
                     "off_streak", "updates", "observations", "calibration_ms",
                     "calibration_model_evaluations"):
            self.assertEqual(getattr(model, attr), 0)
        self.assertEqual(model.raw_scale, 1.0)
        self.assertEqual(model.current_scale, 1.0)
        self.assertEqual(len(model.history), 0)
        for member, original in zip(base.members, before):
            for weight, old in zip(member, original):
                np.testing.assert_array_equal(weight, old)

    def test_parameter_validation(self):
        for kwargs in ({"threshold": 0}, {"threshold": float("nan")},
                       {"threshold": .1, "off_ratio": 1}, {"threshold": .1, "persistence": 0},
                       {"threshold": .1, "min_samples": 6}, {"threshold": .1, "window": 1.5}):
            with self.assertRaises(ValueError):
                GatedCalibratedModel(constant_model(), **kwargs)

    def test_same_history_physics_filter_and_weighted_rollover(self):
        physics = MatchedHistoryPhysicsModel(window=2)
        residual = ResidualCalibratedModel(constant_model(), window=2)
        # Accepted updates have identical targets and weights. Both reject low
        # energy, nonfinite, boundary-clipped and speed-saturated transitions.
        action = np.zeros(2)
        invalid = []
        stopped = moving_state()
        stopped[2:4] = (.1, 0)
        invalid.append((stopped, action, physics_step(stopped, action, 1)))
        wall = moving_state()
        wall[:4] = (.181, 2, -.8, 0)
        invalid.append((wall, action, physics_step(wall, action, 1)))
        fast = moving_state()
        fast[2:4] = (1.8, 0)
        invalid.append((fast, np.ones(2), physics_step(fast, np.ones(2), .1)))
        nan_state = moving_state()
        nan_state[0] = np.nan
        invalid.append((nan_state, action, moving_state()))
        for row in invalid:
            physics.observe(*row)
            residual.observe(*row)
        self.assertEqual(physics.updates, residual.updates)
        self.assertEqual(physics.updates, 0)
        for speed, damping in ((.4, .6), (.8, 1.4), (.5, 2.0)):
            observation(physics, damping, speed)
            observation(residual, damping, speed)
        self.assertEqual(len(physics.history), 2)
        self.assertAlmostEqual(physics.damping, residual.current_scale, places=5)
        self.assertEqual(physics.calibration_model_evaluations, 0)
        physics.reset()
        self.assertEqual(physics.damping, .8)
        self.assertEqual(physics.updates, 0)
        self.assertEqual(len(physics.history), 0)

    def test_same_history_physics_learns_absolute_damping_without_scale_clip(self):
        model = MatchedHistoryPhysicsModel()
        observation(model, 4.0)
        self.assertAlmostEqual(model.damping, 4.0, places=5)
        state, action = moving_state(), np.zeros(2, dtype=np.float32)
        next_state = physics_step(state, action, 4.0)
        np.testing.assert_allclose(model.rollout(state, action[None, None])[0, 0, 1], next_state, atol=1e-6)

    def test_validation_threshold_resets_episode_history_and_filters_rows(self):
        states, next_states, actions, ids = [], [], [], []
        for episode, damping in ((0, 1.0), (1, 1.4)):
            for index in range(5):
                state = moving_state()
                if index == 0:
                    state[2:4] = 0
                action = np.zeros(2, dtype=np.float32)
                states.append(state)
                next_states.append(physics_step(state, action, damping))
                actions.append(action)
                ids.append(episode)
        data = {"states": np.array(states), "actions": np.array(actions),
                "next_states": np.array(next_states), "episode_ids": np.array(ids)}
        threshold, info = gate_threshold_from_validation(constant_model(), data)
        self.assertAlmostEqual(threshold, np.log(1.4), places=5)
        self.assertEqual(info["score_count"], 4)
        self.assertEqual(info["accepted_transitions"], 8)
        self.assertEqual(info["filtered_fraction"], .2)
        self.assertEqual(info["episode_segments"], 2)
        data["next_states"] = np.array([physics_step(s, a, 1.0) for s, a in zip(states, actions)])
        threshold, _ = gate_threshold_from_validation(constant_model(), data)
        self.assertEqual(threshold, .03)

    def test_validation_rejects_empty_or_misaligned_data(self):
        data = {"states": np.zeros((0, 18)), "actions": np.zeros((0, 2)),
                "next_states": np.zeros((0, 18)), "episode_ids": np.zeros(0)}
        with self.assertRaisesRegex(ValueError, "No eligible"):
            gate_threshold_from_validation(constant_model(), data)
        data["actions"] = np.zeros((1, 2))
        with self.assertRaisesRegex(ValueError, "aligned"):
            gate_threshold_from_validation(constant_model(), data)


if __name__ == "__main__":
    unittest.main()
