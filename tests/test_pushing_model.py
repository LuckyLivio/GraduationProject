import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from foresight.pushing_model import (
    HISTORY,
    PushWorldModel,
    apply_targets,
    current_features,
    training_arrays,
    transition_targets,
)


def toy_episodes():
    rng = np.random.default_rng(8371)
    states = rng.normal(0, .2, (2, 11, 6)).astype(np.float32)
    actions = np.empty((2, 10, 3), np.float32)
    actions[..., 0] = rng.integers(0, 4, (2, 10))
    actions[..., 1] = rng.uniform(-.7, .7, (2, 10))
    actions[..., 2] = rng.uniform(.2, .9, (2, 10))
    return states, actions


def transform_states(states, angle, translation):
    transformed = np.asarray(states).copy()
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    transformed[..., :2] = states[..., :2] @ rotation.T + translation
    transformed[..., 3:5] = states[..., 3:5] @ rotation.T
    transformed[..., 2] = (states[..., 2] + angle + np.pi) % (2 * np.pi) - np.pi
    return transformed


class PushingModelTests(unittest.TestCase):
    def test_future_targets_never_enter_current_history(self):
        states, actions = toy_episodes()
        x, history, targets = training_arrays(states, actions)
        changed = states.copy()
        changed[:, 5:] += np.array([5, -3, .4, 2, -4, .3], np.float32)
        changed_x, changed_history, changed_targets = training_arrays(changed, actions)
        # Prediction index 4 may know states 0..4, but cannot know target state 5.
        for index in (4, 14):
            np.testing.assert_array_equal(x[index], changed_x[index])
            np.testing.assert_array_equal(history[index], changed_history[index])
            self.assertGreater(np.linalg.norm(targets[index] - changed_targets[index]), 1)

    def test_history_has_episode_boundaries_and_matches_online_construction(self):
        states, actions = toy_episodes()
        _, history, _ = training_arrays(states, actions)
        model = PushWorldModel()
        for episode in range(2):
            for step in range(10):
                row = history[episode * 10 + step]
                self.assertEqual(int(row[:, -1].sum()), min(step, HISTORY))
                online = model.history_array(states[episode, :step + 1], actions[episode, :step])
                np.testing.assert_array_equal(row, online)
            np.testing.assert_array_equal(history[episode * 10], np.zeros((HISTORY, 16)))

    def test_local_coordinate_targets_round_trip_across_angle_wrap(self):
        states, _ = toy_episodes()
        old, new = states[:, :-1].copy(), states[:, 1:].copy()
        old[0, 0, 2] = np.pi - .01
        new[0, 0, 2] = -np.pi + .02
        target = transition_targets(old, new)
        np.testing.assert_allclose(apply_targets(old, target), new, atol=5e-7)
        self.assertAlmostEqual(float(target[0, 0, 2]), .03, places=5)

    def test_predictions_are_equivariant_under_global_translation_and_rotation(self):
        torch.manual_seed(73)
        states, actions = toy_episodes()
        model = PushWorldModel().eval()
        angle, translation = 1.1, np.array([7., -4.])
        moved = transform_states(states, angle, translation)
        before = model.predict_batch(states[:, 6], actions[:, 6],
            np.stack([model.history_array(s[:7], a[:6]) for s, a in zip(states, actions)]))
        after = model.predict_batch(moved[:, 6], actions[:, 6],
            np.stack([model.history_array(s[:7], a[:6]) for s, a in zip(moved, actions)]))
        np.testing.assert_allclose(after, transform_states(before, angle, translation), atol=2e-6)
        np.testing.assert_allclose(current_features(states[:, 6], actions[:, 6]),
            current_features(moved[:, 6], actions[:, 6]), atol=1e-6)

    def test_open_loop_rollout_never_treats_imagination_as_observed_history(self):
        states, actions = toy_episodes()
        model = PushWorldModel().eval()
        expected = model.history_array(states[0, :7], actions[0, :6])
        observed_contexts = []

        def prediction(current, action, history):
            observed_contexts.append(history.copy())
            next_state = current.copy()
            next_state[:, 0] += 10  # Easily distinguish imagined and real motion.
            return next_state

        with patch.object(model, "predict_batch", side_effect=prediction):
            rollout = model.rollout(states[0, 6], actions[0, 6:10], states[0, :7], actions[0, :6])
        self.assertEqual(rollout.shape, (5, 6))
        self.assertEqual(len(observed_contexts), 4)
        for history in observed_contexts:
            np.testing.assert_array_equal(history, expected)
        self.assertAlmostEqual(float(rollout[-1, 0] - rollout[0, 0]), 40., places=5)

    def test_checkpoint_restores_predictions_and_training_normalization(self):
        torch.manual_seed(73)
        states, actions = toy_episodes()
        x, _, y = training_arrays(states, actions)
        model = PushWorldModel(width=24, context=12).eval()
        model.set_normalization(x, y)
        history = model.history_array(states[0, :7], actions[0, :6])
        before = model.predict_batch(states[:1, 6], actions[:1, 6], history)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path, {"seed": 73, "purpose": "round_trip_test"})
            restored = PushWorldModel.load(path)
            after = restored.predict_batch(states[:1, 6], actions[:1, 6], history)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(restored.width, 24)
        self.assertEqual(restored.context, 12)

    def test_portable_npz_restores_learned_weights_for_both_model_variants(self):
        torch.manual_seed(91)
        states, actions = toy_episodes()
        x, history, y = training_arrays(states, actions)
        # A real optimization step makes this a trained-state round trip,
        # including learned GRU state and normalization rather than init only.
        for use_history, width in [(True, 24), (False, 198)]:
            with self.subTest(use_history=use_history):
                model = PushWorldModel(use_history, width=width, context=12)
                model.set_normalization(x, y)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                normalized_y = (torch.from_numpy(y) - model.y_mean) / model.y_std
                loss = (model(torch.from_numpy(x), torch.from_numpy(history)) - normalized_y).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.eval()
                before = model.predict_batch(states[:1, 6], actions[:1, 6], history[6])
                original_weights = {name: value.detach().cpu().numpy().copy()
                                    for name, value in model.state_dict().items()}
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.npz"
                    np.savez_compressed(path, **original_weights)
                    restored = PushWorldModel.load(path)
                    after = restored.predict_batch(states[:1, 6], actions[:1, 6], history[6])
                np.testing.assert_array_equal(before, after)
                self.assertEqual(restored.use_history, use_history)
                self.assertEqual(restored.width, width)
                if use_history:
                    self.assertEqual(restored.context, 12)
                for name, value in restored.state_dict().items():
                    np.testing.assert_array_equal(value.numpy(), original_weights[name])


if __name__ == "__main__":
    unittest.main()
