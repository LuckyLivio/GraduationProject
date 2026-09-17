"""History-only system identification for the contact-pushing pilot.

This is a deliberately strong privileged-structure baseline: it knows the
simulator's public equations, but fits its unknown parameters using exactly
the observed transitions supplied to it. It never receives the environment's
true parameters. The fitted values are prediction parameters, not proof that
physical parameters are identifiable from a particular set of pushes.
"""

from __future__ import annotations

from itertools import product
from time import perf_counter
from typing import Iterable

import numpy as np

from foresight.pushing_env import PushParams, simulate_push


_LOWER = np.array([.05, .1, -.07, -.07], dtype=np.float64)
_UPPER = np.array([.95, 1., .07, .07], dtype=np.float64)
_NOMINAL = (_LOWER + _UPPER) / 2
# Convert angular error into displacement of a point 25 cm from the centre.
# Velocity residuals count as displacement accumulated over 0.15 seconds.
_ERROR_SCALE = np.array([1., 1., .25, .15, .15, .25 * .15])


def wrap_angle(angle):
    """Return signed angular differences in [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _vector(params: PushParams) -> np.ndarray:
    return np.array([params.damping, params.friction, params.cog_x, params.cog_y])


def _params(values) -> PushParams:
    return PushParams(*map(float, values))


def parameter_candidates() -> list[PushParams]:
    """Return 97 deterministic hypotheses spanning the public search domain.

    The 3**4 grid includes nominal and boundary values. Sixteen fixed random
    hypotheses reduce regular-grid artifacts; this RNG is independent of all
    data-generation and evaluation seeds. Damping extends to 0.95 and COG to
    +/-0.07 m, deliberately covering development shifts outside training.
    """
    axes = [np.linspace(lo, hi, 3) for lo, hi in zip(_LOWER, _UPPER)]
    grid = [_params(values) for values in product(*axes)]
    rng = np.random.default_rng(71931)
    return grid + [_params(values) for values in rng.uniform(_LOWER, _UPPER, (16, 4))]


def _history(history_states, history_actions):
    states = np.asarray(history_states, dtype=np.float64)
    actions = np.asarray(history_actions, dtype=np.float64)
    if actions.size == 0:
        actions = np.empty((0, 3), dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError("history_states must have shape (H + 1, 6)")
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError("history_actions must have shape (H, 3)")
    if len(states) != len(actions) + 1:
        raise ValueError("history must contain one more state than actions")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("history must contain finite observations and actions")
    return states, actions


def identify_parameters(
    history_states,
    history_actions,
    candidates: Iterable[PushParams] | None = None,
) -> tuple[PushParams, dict]:
    """Fit public physics to observed transitions, without hidden-state access.

    Each candidate starts from every observed state, so fitting measures
    one-step prediction error rather than accumulated rollout drift. With the
    default candidate bank, two coordinate-search rounds refine the best fit.
    Supplying candidates disables refinement and uses only those hypotheses.
    No history returns nominal parameters and an explicit no-evidence status.
    """
    started = perf_counter()
    states, actions = _history(history_states, history_actions)
    supplied_candidates = candidates is not None
    bank = list(candidates) if supplied_candidates else parameter_candidates()
    if not bank:
        raise ValueError("at least one parameter candidate is required")
    for params in bank:
        values = _vector(params)
        if not np.isfinite(values).all():
            raise ValueError("parameter candidates must be finite")
        if np.any(values < _LOWER) or np.any(values > _UPPER):
            raise ValueError("parameter candidate is outside the public parameter domain")
    if not len(actions):
        estimate = min(bank, key=lambda params: np.linalg.norm((_vector(params) - _NOMINAL) / (_UPPER - _LOWER)))
        return estimate, {
            "status": "no_history",
            "history_steps": 0,
            "candidate_count": len(bank),
            "hypotheses_evaluated": 0,
            "model_calls": 0,
            "fit_rmse": None,
            "elapsed_ms": (perf_counter() - started) * 1000,
        }

    calls = 0
    cached: dict[tuple, float] = {}

    def objective(params):
        nonlocal calls
        key = tuple(_vector(params))
        if key not in cached:
            residuals = []
            for state, action, observed_next in zip(states[:-1], actions, states[1:]):
                predicted, _ = simulate_push(state.copy(), action.copy(), params)
                residual = np.asarray(predicted, dtype=np.float64) - observed_next
                residual[2] = wrap_angle(residual[2])
                residuals.append(residual * _ERROR_SCALE)
                calls += 1
            cached[key] = float(np.mean(np.sum(np.square(residuals), axis=1)))
        return cached[key]

    scores = [objective(params) for params in bank]
    best_index = int(np.argmin(scores))
    estimate, best_score = bank[best_index], scores[best_index]
    if not supplied_candidates:
        step = (_UPPER - _LOWER) / 4
        for _ in range(2):
            for dimension in range(4):
                centre = _vector(estimate)
                for direction in (-1., 1.):
                    trial = centre.copy()
                    trial[dimension] = np.clip(trial[dimension] + direction * step[dimension], _LOWER[dimension], _UPPER[dimension])
                    proposal = _params(trial)
                    score = objective(proposal)
                    if score < best_score:
                        estimate, best_score = proposal, score
            step /= 2

    return estimate, {
        "status": "fitted",
        "history_steps": len(actions),
        "candidate_count": len(bank),
        "hypotheses_evaluated": len(cached),
        "model_calls": calls,
        "fit_rmse": float(np.sqrt(best_score)),
        "initial_best_rmse": float(np.sqrt(min(scores))),
        "elapsed_ms": (perf_counter() - started) * 1000,
    }


def predict_physics(state, actions, params: PushParams) -> np.ndarray:
    """Roll out a sequence using the fitted parameters, including its start."""
    current = np.asarray(state, dtype=np.float64).copy()
    sequence = np.asarray(actions, dtype=np.float64)
    if current.shape != (6,) or not np.isfinite(current).all():
        raise ValueError("state must be a finite vector of length 6")
    if sequence.size == 0:
        sequence = np.empty((0, 3), dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[1] != 3 or not np.isfinite(sequence).all():
        raise ValueError("actions must have shape (H, 3) and be finite")
    predicted = [current.copy()]
    for action in sequence:
        current, _ = simulate_push(current, action, params)
        predicted.append(np.asarray(current, dtype=np.float64).copy())
    return np.stack(predicted)


def goal_cost(states, goal, *, angle_weight: float = .25):
    """Squared terminal pose error for one rollout or a batch of rollouts.

    Input shape is (..., T, 6), goal contains at least [x, y, theta]. The
    default angular scale represents 25 cm of corner displacement per radian.
    This cost has no access to hidden parameters or future ground truth.
    """
    trajectories = np.asarray(states, dtype=np.float64)
    target = np.asarray(goal, dtype=np.float64)
    if trajectories.ndim < 2 or trajectories.shape[-1] != 6 or trajectories.shape[-2] < 1:
        raise ValueError("states must have shape (..., T, 6), with T >= 1")
    if target.ndim != 1 or len(target) < 3:
        raise ValueError("goal must contain x, y and theta")
    if not np.isfinite(trajectories).all() or not np.isfinite(target[:3]).all():
        raise ValueError("states and target pose must be finite")
    if not np.isfinite(angle_weight) or angle_weight < 0:
        raise ValueError("angle_weight must be finite and non-negative")
    final = trajectories[..., -1, :]
    position_error = np.sum(np.square(final[..., :2] - target[:2]), axis=-1)
    rotation_error = wrap_angle(final[..., 2] - target[2])
    cost = position_error + np.square(angle_weight * rotation_error)
    return float(cost) if np.ndim(cost) == 0 else cost
