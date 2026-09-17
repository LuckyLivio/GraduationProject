"""Online multiplicative calibration of a frozen spatial dynamics model.

This exploratory mechanism uses only previously observed transitions.  A scalar
least-squares fit corrects the learned damping field without changing any base
network weights.  It deliberately shares history access with the online physics
baseline; it is not a new training run or a claim of algorithmic novelty.
"""
from __future__ import annotations

from collections import deque
import time

import numpy as np

from .env import physics_step
from .model import transition_targets


class ResidualCalibratedModel:
    """Frozen learned spatial prior with a five-valid-transition scale fit.

    For observed damping ``d_i``, frozen mean prediction ``p_i`` and velocity
    energy ``w_i``, scale = sum(w_i*p_i*d_i) / sum(w_i*p_i**2), clipped to
    [0.5, 2.5]. Predictions for a past position are cached when first observed,
    so each accepted transition adds only ``n_members`` MLP evaluations.

    ``calibration_model_evaluations`` counts those member-position evaluations
    separately from rollout member-transitions. ``calibration_ms`` measures all
    observe calls, including validation of rejected transitions. Neither cost is
    included in MPC's rollout budget, so experiments must report both separately.
    """

    def __init__(self, base, window: int = 5):
        if int(window) != window or window < 1:
            raise ValueError("window must be a positive integer")
        self.base = base
        self.n_members = int(base.n_members)
        if self.n_members < 1:
            raise ValueError("base model must have at least one member")
        self.window = int(window)
        self.history = deque(maxlen=self.window)
        self.reset()

    def reset(self):
        """Clear episode-local calibration without mutating the frozen base."""
        self.history.clear()
        self.current_scale = 1.0
        self.updates = 0
        self.observations = 0
        self.calibration_model_evaluations = 0
        self.calibration_ms = 0.0
        self.last_calibration_model_evaluations = 0
        self.last_calibration_ms = 0.0

    def observe(self, state, action, next_state):
        started = time.perf_counter()
        self.observations += 1
        self.last_calibration_model_evaluations = 0
        try:
            state = np.asarray(state, dtype=np.float32)
            action = np.asarray(action, dtype=np.float32)
            next_state = np.asarray(next_state, dtype=np.float32)
            if state.shape != (18,) or next_state.shape != (18,) or action.shape != (2,):
                raise ValueError("Expected state[18], action[2], next_state[18]")
            if not all(np.all(np.isfinite(x)) for x in (state, action, next_state)):
                return
            positions, targets = transition_targets({
                "states": state[None], "actions": np.clip(action, -1, 1)[None],
                "next_states": next_state[None],
            })
            if len(targets) == 0:
                return
            member_positions = np.broadcast_to(positions, (self.n_members, 1, 2))
            predicted = np.asarray(self.base.damping(member_positions), dtype=np.float64)
            self.last_calibration_model_evaluations = self.n_members
            self.calibration_model_evaluations += self.n_members
            prior = float(predicted.mean())
            if not np.isfinite(prior) or prior <= 1e-8:
                return
            weight = float(np.sum(state[2:4] ** 2))
            self.history.append({"position": positions[0].copy(),
                                 "observed_damping": float(targets[0]),
                                 "prior_damping": prior, "weight": weight})
            numerator = sum(item["weight"] * item["prior_damping"] * item["observed_damping"] for item in self.history)
            denominator = sum(item["weight"] * item["prior_damping"] ** 2 for item in self.history)
            self.current_scale = float(np.clip(numerator / denominator, .5, 2.5))
            self.updates += 1
        finally:
            self.last_calibration_ms = float((time.perf_counter() - started) * 1000)
            self.calibration_ms += self.last_calibration_ms

    def damping(self, positions):
        return self.base.damping(positions) * self.current_scale

    def rollout(self, state, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 3 or actions.shape[-1] != 2:
            raise ValueError("actions must have shape [candidates, horizon, 2]")
        count, horizon = actions.shape[:2]
        states = np.broadcast_to(np.asarray(state, dtype=np.float32), (self.n_members, count, 18)).copy()
        result = np.empty((self.n_members, count, horizon + 1, 18), dtype=np.float32)
        result[:, :, 0] = states
        for step in range(horizon):
            states = physics_step(states, actions[None, :, step], self.damping(states[..., :2]))
            result[:, :, step + 1] = states
        return result
