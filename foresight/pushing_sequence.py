"""Differentiable sequence training for the existing object world model.

The physics context contains only real observations available before the
rollout. It remains fixed while the model imagines future states; an imagined
transition must never become evidence about the object's hidden properties.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from foresight.pushing_model import HISTORY


def current_features_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Return local velocity, contact-face one-hot and push controls: [..., 9].

    This is the differentiable equivalent of ``pushing_model.current_features``.
    Floating state/action inputs retain their dtype and device. The discrete
    face index, as in the NumPy implementation, is not differentiable.
    """
    if states.shape[-1:] != (6,) or actions.shape != (*states.shape[:-1], 3):
        raise ValueError("Expected states [..., 6] and matching actions [..., 3].")
    c, sn = torch.cos(states[..., 2]), torch.sin(states[..., 2])
    velocity = torch.stack((c * states[..., 3] + sn * states[..., 4],
                            -sn * states[..., 3] + c * states[..., 4],
                            states[..., 5]), dim=-1)
    faces = F.one_hot(actions[..., 0].long(), num_classes=4).to(dtype=states.dtype)
    return torch.cat((velocity, faces, actions[..., 1:]), dim=-1)


def apply_targets_torch(states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Convert local pose increments and next velocities to world states.

    Targets contain local delta-x, delta-y, delta-angle, then *next* local
    velocities. Velocities are not increments. Angle wrapping matches the
    NumPy implementation and keeps derivatives away from its wrap boundary.
    """
    if states.shape[-1:] != (6,) or targets.shape != states.shape:
        raise ValueError("Expected matching states and targets with shape [..., 6].")
    c, sn = torch.cos(states[..., 2]), torch.sin(states[..., 2])
    angle = torch.remainder(states[..., 2] + targets[..., 2] + torch.pi,
                            2 * torch.pi) - torch.pi
    return torch.stack((states[..., 0] + c * targets[..., 0] - sn * targets[..., 1],
                        states[..., 1] + sn * targets[..., 0] + c * targets[..., 1],
                        angle,
                        c * targets[..., 3] - sn * targets[..., 4],
                        sn * targets[..., 3] + c * targets[..., 4],
                        targets[..., 5]), dim=-1)


def rollout_torch(model, initial: torch.Tensor, actions: torch.Tensor,
                  history: torch.Tensor,
                  teacher_states: torch.Tensor | None = None) -> torch.Tensor:
    """Predict [B, H+1, 6], preserving gradients through every future step.

    By default each transition receives the previous prediction. Supplying
    ``teacher_states`` of shape [B, H+1, 6] explicitly selects teacher forcing:
    step t receives teacher_states[:, t], never its future target at t+1.
    Both modes keep the observed history fixed and return ``initial`` first.
    Teacher forcing is a training control, not an open-loop evaluation.
    """
    if initial.ndim != 2 or initial.shape[-1] != 6:
        raise ValueError("Expected initial states with shape [B, 6].")
    batch = initial.shape[0]
    if actions.ndim != 3 or actions.shape[0] != batch or actions.shape[-1] != 3:
        raise ValueError("Expected actions with shape [B, H, 3].")
    horizon = actions.shape[1]
    if history.shape != (batch, HISTORY, 16):
        raise ValueError(f"Expected frozen history with shape [B, {HISTORY}, 16].")
    if teacher_states is not None and teacher_states.shape != (batch, horizon + 1, 6):
        raise ValueError("Expected teacher states with shape [B, H+1, 6].")

    predicted = [initial]
    for step in range(horizon):
        state = predicted[-1] if teacher_states is None else teacher_states[:, step]
        features = current_features_torch(state, actions[:, step])
        normalized_targets = model(features, history)
        targets = normalized_targets * model.y_std + model.y_mean
        predicted.append(apply_targets_torch(state, targets))
    return torch.stack(predicted, dim=1)


def pose_loss(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Mean over samples, future steps and four normalized error terms.

    Excludes the supplied initial state. The four terms are squared position
    distance / 0.25**2, 2*(1-cos(angle error)) / 0.5**2, squared linear velocity
    distance / 0.5**2, and squared angular velocity error / 1.0**2. They are
    averaged over the four components. The periodic angle loss
    treats equivalent orientations on opposite sides of pi consistently.
    """
    if (predicted.ndim != 3 or predicted.shape[-1] != 6 or
            predicted.shape[1] < 2 or truth.shape != predicted.shape):
        raise ValueError("Expected matching [B, H+1, 6] paths with H >= 1.")
    delta = predicted[:, 1:] - truth[:, 1:]
    position = delta[..., :2].square().sum(dim=-1) / .25**2
    angle = 2 * (1 - torch.cos(delta[..., 2])) / .5**2
    velocity = delta[..., 3:5].square().sum(dim=-1) / .5**2
    angular_velocity = delta[..., 5].square() / 1.0**2
    return (position + angle + velocity + angular_velocity).mean() / 4
