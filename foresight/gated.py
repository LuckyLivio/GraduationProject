"""Persistence-gated calibration using only previously observed transitions.

The gate retains a frozen learned prior when its rolling fitted scale is near
one. It is a heuristic change detector, not a calibrated probability or a claim
of novelty. Rejected transitions do not advance persistence or expire history;
long periods without identifiable motion can therefore leave a stale estimate.
"""
from __future__ import annotations

from collections import deque
import time

import numpy as np

from .model import transition_targets
from .planner import PhysicsPredictor
from .residual import ResidualCalibratedModel


def _positive_integer(value, name):
    if int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class GatedCalibratedModel(ResidualCalibratedModel):
    """Apply the common residual fit only after persistent scale disagreement.

    ``raw_scale`` is the candidate fit before gate application, including the
    superclass's [0.5, 2.5] clipping. The score is ``abs(log(raw_scale))``. After
    ``min_samples`` accepted observations, ``persistence`` consecutive accepted
    scores above ``threshold`` activate calibration. The same number below
    ``off_ratio * threshold`` restore the frozen prior. Scores in between reset
    the relevant streak; invalid observations leave both streaks unchanged.

    At the defaults, activation first becomes possible on accepted observation
    five. Inactive predictions are exactly those of the frozen base. Gate work
    is included in calibration timing; no additional network calls are made.
    """

    def __init__(self, base, threshold: float, window: int = 5,
                 persistence: int = 3, min_samples: int = 3,
                 off_ratio: float = .6):
        self.threshold = float(threshold)
        self.gate_threshold = self.threshold
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        self.persistence = _positive_integer(persistence, "persistence")
        self.min_samples = _positive_integer(min_samples, "min_samples")
        if self.min_samples > window:
            raise ValueError("min_samples cannot exceed window")
        self.off_ratio = float(off_ratio)
        if not np.isfinite(self.off_ratio) or not 0 < self.off_ratio < 1:
            raise ValueError("off_ratio must be strictly between zero and one")
        super().__init__(base, window=window)

    def reset(self):
        super().reset()
        self.raw_scale = 1.0
        self.gate_score = 0.0
        self.gate_active = False
        self.gate_on_events = 0
        self.gate_off_events = 0
        self.on_streak = 0
        self.off_streak = 0

    def observe(self, state, action, next_state):
        started = time.perf_counter()
        previous_updates = self.updates
        try:
            super().observe(state, action, next_state)
            if self.updates == previous_updates:
                return
            self.raw_scale = self.current_scale
            self.gate_score = float(abs(np.log(self.raw_scale)))
            if len(self.history) >= self.min_samples:
                if self.gate_active:
                    self.on_streak = 0
                    self.off_streak = (self.off_streak + 1
                                       if self.gate_score < self.off_ratio * self.threshold else 0)
                    if self.off_streak >= self.persistence:
                        self.gate_active = False
                        self.gate_off_events += 1
                        self.off_streak = 0
                else:
                    self.off_streak = 0
                    self.on_streak = (self.on_streak + 1
                                      if self.gate_score > self.threshold else 0)
                    if self.on_streak >= self.persistence:
                        self.gate_active = True
                        self.gate_on_events += 1
                        self.on_streak = 0
            self.current_scale = self.raw_scale if self.gate_active else 1.0
        finally:
            elapsed = float((time.perf_counter() - started) * 1000)
            # The superclass already added its own time, including rejected rows.
            self.calibration_ms += max(0.0, elapsed - self.last_calibration_ms)
            self.last_calibration_ms = elapsed


class MatchedHistoryPhysicsModel(PhysicsPredictor):
    """Local absolute-damping WLS with the residual model's filter and history.

    This stronger physical baseline gets the same last ``window`` accepted
    transitions and energy weights, but does not use a learned spatial prior.
    Its estimate is ``sum(w_i * observed_damping_i) / sum(w_i)``, clipped to
    [0.02, 6]. Unlike scale clipping, this bounds absolute physical damping.
    There are no neural evaluations; all observation work is timed separately.
    """

    def __init__(self, damping: float = .8, window: int = 5):
        window = _positive_integer(window, "window")
        if not np.isfinite(damping) or not .02 <= damping <= 6:
            raise ValueError("initial damping must be finite and in [0.02, 6]")
        super().__init__(damping=damping, adaptive=True)
        self.window = window
        self.history = deque(maxlen=window)
        self.reset()

    def reset(self):
        super().reset()
        self.updates = 0
        self.observations = 0
        self.calibration_model_evaluations = 0
        self.last_calibration_model_evaluations = 0
        self.calibration_ms = 0.0
        self.last_calibration_ms = 0.0

    def observe(self, state, action, next_state):
        started = time.perf_counter()
        self.observations += 1
        try:
            state = np.asarray(state, dtype=np.float32)
            action = np.asarray(action, dtype=np.float32)
            next_state = np.asarray(next_state, dtype=np.float32)
            if state.shape != (18,) or next_state.shape != (18,) or action.shape != (2,):
                raise ValueError("Expected state[18], action[2], next_state[18]")
            if not all(np.all(np.isfinite(x)) for x in (state, action, next_state)):
                return
            _, targets = transition_targets({
                "states": state[None], "actions": np.clip(action, -1, 1)[None],
                "next_states": next_state[None],
            })
            if len(targets) == 0:
                return
            weight = float(np.sum(state[2:4] ** 2))
            self.history.append((float(targets[0]), weight))
            self.damping = float(np.clip(
                sum(d * w for d, w in self.history) / sum(w for _, w in self.history), .02, 6.0))
            self.updates += 1
        finally:
            self.last_calibration_ms = float((time.perf_counter() - started) * 1000)
            self.calibration_ms += self.last_calibration_ms


def gate_threshold_from_validation(base, data, quantile: float = .99,
                                   window: int = 5, min_samples: int = 3):
    """Fit a frozen threshold on ordered nominal-validation episodes only.

    Return ``(threshold, metadata)``. Rows must already be chronological within
    each contiguous episode block. The rolling history resets at every episode
    boundary, and only fits with at least ``min_samples`` accepted rows count.
    The caller is responsible for reserving nominal validation data separately
    from training and evaluation; this function cannot infer dataset provenance.
    """
    if not np.isfinite(quantile) or not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    min_samples = _positive_integer(min_samples, "min_samples")
    if min_samples > window:
        raise ValueError("min_samples cannot exceed window")
    states = np.asarray(data["states"])
    actions = np.asarray(data["actions"])
    next_states = np.asarray(data["next_states"])
    episode_ids = np.asarray(data["episode_ids"])
    count = len(states)
    if (states.shape != (count, 18) or actions.shape != (count, 2)
            or next_states.shape != (count, 18) or episode_ids.shape != (count,)):
        raise ValueError("Expected aligned states[N,18], actions[N,2], next_states[N,18], episode_ids[N]")
    model = ResidualCalibratedModel(base, window=window)
    scores = []
    accepted = 0
    episode_segments = 0
    for index in range(count):
        if index == 0 or episode_ids[index] != episode_ids[index - 1]:
            model.reset()
            episode_segments += 1
        before = model.updates
        model.observe(states[index], actions[index], next_states[index])
        if model.updates == before:
            continue
        accepted += 1
        if len(model.history) >= min_samples:
            scores.append(float(abs(np.log(model.current_scale))))
    if not scores:
        raise ValueError("No eligible nominal-validation scores; need enough valid transitions per episode")
    observed_quantile = float(np.quantile(scores, quantile))
    threshold = max(.03, observed_quantile)
    return threshold, {
        "threshold": threshold, "quantile": float(quantile),
        "observed_quantile": observed_quantile, "threshold_floor": .03,
        "score": "abs(log(clipped_scale))", "window": int(window),
        "min_samples": min_samples, "score_count": len(scores), "n": len(scores),
        "transitions": count, "accepted_transitions": accepted,
        "filtered_fraction": (count - accepted) / count,
        "episode_segments": episode_segments,
        "provenance_requirement": "nominal validation only; no evaluation threshold tuning",
    }
