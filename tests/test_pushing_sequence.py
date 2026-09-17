"""Numerical equivalence, causal evidence and gradients for sequence training."""
import unittest

import numpy as np
import torch

from foresight.pushing_model import (
    HISTORY, PushWorldModel, apply_targets, current_features, training_arrays,
)
from foresight.pushing_sequence import (
    apply_targets_torch, current_features_torch, pose_loss, rollout_torch,
)


def sample_batch():
    rng = np.random.default_rng(891)
    states = rng.normal(0, .3, (3, 12, 6)).astype(np.float32)
    states[..., 2] *= 6
    actions = rng.uniform(-.6, .6, (3, 11, 3)).astype(np.float32)
    actions[..., 0] = rng.integers(0, 4, (3, 11))
    actions[..., 2] = np.abs(actions[..., 2]) + .2
    return states, actions


def prepared_model(use_history=True):
    torch.manual_seed(93)
    states, actions = sample_batch()
    model = PushWorldModel(use_history, width=24, context=12).eval()
    x, _, y = training_arrays(states, actions)
    model.set_normalization(x, y)
    history = np.stack([model.history_array(s[:7], a[:6])
                        for s, a in zip(states, actions)])
    return model, torch.from_numpy(states[:, 6]), torch.from_numpy(actions[:, 6:]), torch.from_numpy(history)


class PushingSequenceTests(unittest.TestCase):
    def test_torch_features_and_targets_match_numpy_for_batched_paths(self):
        states, actions = sample_batch()
        states = states[:, :-1]
        states[0, 0, 2], states[1, 0, 2] = np.pi - .01, -np.pi + .01
        targets = np.random.default_rng(14).normal(0, .3, states.shape).astype(np.float32)
        np.testing.assert_allclose(
            current_features_torch(torch.from_numpy(states), torch.from_numpy(actions)).numpy(),
            current_features(states, actions), atol=3e-7)
        np.testing.assert_allclose(
            apply_targets_torch(torch.from_numpy(states), torch.from_numpy(targets)).numpy(),
            apply_targets(states, targets), atol=8e-7)

    def test_rollout_matches_existing_numpy_inference_with_and_without_memory(self):
        for use_history in (True, False):
            with self.subTest(use_history=use_history):
                model, initial, actions, history = prepared_model(use_history)
                expected = [initial.numpy()]
                for action in actions.numpy().transpose(1, 0, 2):
                    expected.append(model.predict_batch(expected[-1], action, history.numpy()))
                expected = np.stack(expected, axis=1)
                actual = rollout_torch(model, initial, actions, history).detach().numpy()
                np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_open_loop_uses_predictions_and_frozen_observed_history(self):
        model, initial, actions, history = prepared_model()
        original = history.clone()
        seen = []
        hook = model.register_forward_pre_hook(
            lambda _model, inputs: seen.append((inputs[0].detach().clone(), inputs[1].detach().clone())))
        try:
            predicted = rollout_torch(model, initial, actions, history)
        finally:
            hook.remove()
        self.assertEqual(len(seen), actions.shape[1])
        for step, (features, context) in enumerate(seen):
            torch.testing.assert_close(features, current_features_torch(predicted[:, step], actions[:, step]))
            torch.testing.assert_close(context, original, rtol=0, atol=0)
        torch.testing.assert_close(history, original, rtol=0, atol=0)
        # Later commands cannot affect an already predicted prefix.
        changed_actions = actions.clone()
        changed_actions[:, 3:, 2] += 1
        changed = rollout_torch(model, initial, changed_actions, history)
        torch.testing.assert_close(changed[:, :4], predicted[:, :4], rtol=0, atol=0)

    def test_teacher_mode_uses_only_current_truth_and_never_future_target(self):
        model, initial, actions, history = prepared_model()
        states, _ = sample_batch()
        teacher = torch.from_numpy(states[:, 6:])
        predicted = rollout_torch(model, initial, actions, history, teacher_states=teacher)
        expected = [initial]
        for step in range(actions.shape[1]):
            state = teacher[:, step]
            targets = model(current_features_torch(state, actions[:, step]), history)
            expected.append(apply_targets_torch(state, targets * model.y_std + model.y_mean))
        torch.testing.assert_close(predicted, torch.stack(expected, dim=1), rtol=0, atol=0)
        poisoned_target = teacher.clone()
        poisoned_target[:, -1] = float("nan")
        no_leakage = rollout_torch(model, initial, actions, history, teacher_states=poisoned_target)
        torch.testing.assert_close(no_leakage, predicted, rtol=0, atol=0)
        changed = teacher.clone()
        changed[:, 2] += 10
        one_changed_step = rollout_torch(model, initial, actions, history, teacher_states=changed)
        torch.testing.assert_close(one_changed_step[:, :3], predicted[:, :3], rtol=0, atol=0)
        torch.testing.assert_close(one_changed_step[:, 4:], predicted[:, 4:], rtol=0, atol=0)
        self.assertGreater(float((one_changed_step[:, 3] - predicted[:, 3]).detach().abs().max()), 1)

    def test_final_step_loss_backpropagates_through_first_prediction_and_memory(self):
        model, initial, actions, history = prepared_model()
        outputs = []

        def record_output(_model, _inputs, output):
            output.retain_grad()
            outputs.append(output)

        hook = model.register_forward_hook(record_output)
        try:
            predicted = rollout_torch(model, initial, actions, history)
        finally:
            hook.remove()
        # No intermediate losses: this specifically detects detached rollouts.
        predicted[:, -1, :2].square().sum().backward()
        self.assertIsNotNone(outputs[0].grad)
        self.assertGreater(float(outputs[0].grad[:, :2].abs().sum()), 0)
        self.assertGreater(float(model.memory.weight_ih_l0.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(outputs[0].grad).all())

    def test_batch_members_cannot_share_history_or_recurrent_state(self):
        model, initial, actions, history = prepared_model()
        batch = rollout_torch(model, initial, actions, history)
        for index in range(len(initial)):
            alone = rollout_torch(model, initial[index:index+1], actions[index:index+1], history[index:index+1])
            torch.testing.assert_close(alone[0], batch[index], rtol=2e-5, atol=2e-6)
        changed_history = history.clone()
        changed_history[1, :, :15] += 30
        changed = rollout_torch(model, initial, actions, changed_history)
        torch.testing.assert_close(changed[[0, 2]], batch[[0, 2]], rtol=0, atol=0)
        self.assertGreater(float((changed[1] - batch[1]).detach().abs().max()), 1e-5)

    def test_periodic_loss_and_wrap_have_finite_gradients(self):
        initial = torch.zeros((1, 6), dtype=torch.float64)
        initial[:, 2] = torch.pi - .01
        targets = torch.tensor([[.1, .2, .03, .2, .1, .4]], dtype=torch.float64, requires_grad=True)
        result = apply_targets_torch(initial, targets)
        self.assertAlmostEqual(float(result[0, 2].detach()), -np.pi + .02, places=12)
        result.sum().backward()
        self.assertTrue(torch.isfinite(targets.grad).all())
        self.assertAlmostEqual(float(targets.grad[0, 2]), 1., places=12)
        truth = torch.zeros((1, 2, 6), dtype=torch.float64)
        truth[0, 1, 2] = -torch.pi + .01
        prediction = truth.clone()
        prediction[0, 0] = 100  # The supplied initial state is excluded.
        prediction[0, 1, 2] = torch.pi - .01
        prediction.requires_grad_()
        loss = pose_loss(prediction, truth)
        self.assertAlmostEqual(float(loss.detach()), 2 * (1 - np.cos(.02)) / .5**2 / 4, places=12)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        torch.testing.assert_close(prediction.grad[:, 0], torch.zeros_like(prediction.grad[:, 0]))

    def test_loss_averages_four_terms_samples_and_future_steps(self):
        truth = torch.zeros(2, 3, 6, dtype=torch.float64)
        predicted = truth.clone()
        predicted[:, 1:, 0] = .25
        predicted[:, 1:, 3] = .5
        predicted[:, 1:, 5] = 1.
        self.assertAlmostEqual(float(pose_loss(predicted, truth)), .75, places=12)
        predicted[:, 1:, 2] = torch.pi
        self.assertAlmostEqual(float(pose_loss(predicted, truth)), 4.75, places=12)

    def test_shape_checks_prevent_accidental_batch_broadcasting(self):
        model, initial, actions, history = prepared_model()
        with self.assertRaises(ValueError):
            rollout_torch(model, initial, actions, history[:1])
        with self.assertRaises(ValueError):
            rollout_torch(model, initial, actions[:1], history)
        with self.assertRaises(ValueError):
            rollout_torch(model, initial, actions, history,
                          teacher_states=torch.zeros(1, actions.shape[1]+1, 6))
        with self.assertRaises(ValueError):
            pose_loss(initial[:, None], initial[:, None])
        empty = rollout_torch(model, initial, actions[:, :0], history)
        torch.testing.assert_close(empty, initial[:, None])


if __name__ == "__main__":
    unittest.main()
