"""Paired demonstrations that separate open-loop prediction from navigation.

The diagnostic uses a declared cross-field motion preset, never a search for a
winning scene. Future actions are fixed before any predictor sees them. Learned
weights and calibration threshold come from the existing frozen study. Hidden
damping is used only by the simulator and the explicitly labelled field display.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .env import DT, NavigationEnv, damping_at
from .gated import GatedCalibratedModel, MatchedHistoryPhysicsModel
from .planner import MPCPlanner, PhysicsPredictor


METHOD_IDS = ('global_physics', 'local_identification', 'frozen_learned', 'gated_calibrated')
CONDITIONS = {'nominal': 1.0, 'global_shift': 1.7}
WARMUP_STEPS = 8
WARMUP_ACTION = np.array([.4, 0], dtype=np.float32)


def _models(base, threshold, global_damping):
    # Each comparison has fresh episode state; even the frozen wrappers do not
    # share a mutable object with a live browser session or another request.
    return (
        PhysicsPredictor(damping=global_damping),
        MatchedHistoryPhysicsModel(damping=global_damping),
        copy.deepcopy(base),
        GatedCalibratedModel(copy.deepcopy(base), threshold=threshold),
    )


def _condition_scale(condition):
    if condition not in CONDITIONS:
        raise ValueError('condition must be nominal or global_shift')
    return CONDITIONS[condition]


def _status(model):
    return {
        'gate_active': bool(getattr(model, 'gate_active', False)),
        'gate_score': float(getattr(model, 'gate_score', 0)),
        'gate_threshold': float(getattr(model, 'gate_threshold', 0)),
        'learned_scale': float(getattr(model, 'current_scale', 1)),
        'local_damping': (float(model.damping)
                          if isinstance(model, PhysicsPredictor) else None),
        'updates': int(getattr(model, 'updates', 0)),
    }


def _field_display(base, scale, size=31):
    coordinates = np.linspace(.2, 9.8, size, dtype=np.float32)
    x, y = np.meshgrid(coordinates, coordinates)
    positions = np.stack((x.ravel(), y.ravel()), axis=-1)
    learned = base.damping(np.broadcast_to(positions, (base.n_members, len(positions), 2))).mean(axis=0)
    return {
        'x': coordinates.tolist(), 'y': coordinates.tolist(),
        'truth': damping_at(positions, scale).reshape(size, size).tolist(),
        'learned': learned.reshape(size, size).tolist(),
        'evaluator_only': True, 'units': 'inverse_seconds',
    }


def _future_actions(action_mode, horizon):
    actions = np.zeros((horizon, 2), dtype=np.float32)
    if action_mode == 'cruise':
        actions[:, 0] = .28
    elif action_mode == 'brake':
        # A predeclared short brake pulse, then coast; no future feedback.
        actions[:min(4, horizon), 0] = -.2
    elif action_mode != 'coast':
        raise ValueError('action_mode must be cruise, coast or brake')
    return actions


def build_comparison(base, threshold, global_damping, seed=2026,
                     condition='nominal', action_mode='cruise', horizon=16):
    """Predict a common action suffix, then execute it once without replanning.

    Paths include the snapshot at index zero. Metrics exclude that known point
    and use only executed transitions if the physical episode ends early. Each
    predictor receives the same eight past transitions and no future updates.
    """
    if isinstance(horizon, bool) or int(horizon) != horizon or not 1 <= horizon <= 32:
        raise ValueError('horizon must be an integer from 1 to 32')
    horizon = int(horizon)
    scale = _condition_scale(condition)
    actions = _future_actions(action_mode, horizon)
    env = NavigationEnv(seed=int(seed), scenario='open', damping_scale=scale)
    # A motion diagnostic, separate from the randomly seeded navigation task.
    state = np.array([2, 4.5, 1, 0, 9, 7,
                      3, .8, 0, 0, 5, 9.2, 0, 0, 7, .8, 0, 0], dtype=np.float32)
    env.state = state
    history = [state.copy()]
    predictors = _models(base, threshold, global_damping)
    for _ in range(WARMUP_STEPS):
        next_state, _, terminated, truncated, _ = env.step(WARMUP_ACTION)
        for predictor in predictors:
            predictor.observe(state, WARMUP_ACTION, next_state)
        history.append(next_state.copy())
        state = next_state
        if terminated or truncated:
            raise RuntimeError('Diagnostic warmup ended unexpectedly')
    snapshot = state.copy()
    results = []
    # Compute every prediction before executing any future transition.
    for method_id, predictor in zip(METHOD_IDS, predictors):
        started = time.perf_counter()
        path = predictor.rollout(snapshot, actions[None]).mean(axis=0)[0]
        results.append({
            'id': method_id, 'predicted_path': path.tolist(),
            'prediction_ms': float((time.perf_counter() - started) * 1000),
            **_status(predictor),
        })
    actual = [snapshot.copy()]
    final = {'success': False, 'collision': False, 'timeout': False}
    for action in actions:
        state, _, terminated, truncated, final = env.step(action)
        actual.append(state.copy())
        if terminated or truncated:
            break
    actual = np.asarray(actual)
    executed = len(actual) - 1
    for result in results:
        predicted = np.asarray(result['predicted_path'])[:executed + 1, :2]
        distances = np.linalg.norm(predicted - actual[:, :2], axis=-1)
        result.update({
            'position_errors': distances.tolist(),
            'endpoint_error': float(distances[-1]),
            'position_rmse': float(np.sqrt(np.mean(distances[1:] ** 2))),
            'evaluated_steps': executed,
        })
    return {
        'view': 'prediction', 'seed': int(seed), 'condition': condition,
        'action_mode': action_mode, 'horizon': horizon, 'executed_steps': executed,
        'dt': DT, 'snapshot': snapshot.tolist(),
        'history': np.asarray(history).tolist(),
        'history_actions': np.repeat(WARMUP_ACTION[None], WARMUP_STEPS, axis=0).tolist(),
        'actions': actions.tolist(), 'actual_path': actual.tolist(), 'methods': results,
        'terminal': {key: bool(final[key]) for key in ('success', 'collision', 'timeout')} | {'done': bool(env.done)},
        'field': _field_display(base, scale),
        'metadata': {
            'diagnostic_preset': 'fixed_cross_field_motion',
            'seed_changes_preset': False,
            'warmup_steps': WARMUP_STEPS, 'warmup_action': WARMUP_ACTION.tolist(),
            'model_training_seed': 142, 'global_damping': float(global_damping),
            'rollout_updates': False, 'replanning': False,
            'metric': 'Euclidean position RMSE over executed future steps; excludes known snapshot',
            'terminal_policy': 'Stop truth at first termination; score only matching predicted prefix',
            'field_access': 'Display/evaluator only; hidden truth field never enters a predictor',
            'scope': 'Mechanism diagnostic; not a representative success-rate benchmark',
            'brake_action': 'Four steps of [-0.2,0], then zero action',
        },
    }


def build_navigation(base, threshold, global_damping, seed=2026,
                     condition='nominal', scenario='crossing'):
    """Four closed-loop runs with the same initial state and model-step budget.

    Future actions and visited states diverge between controllers. Consequently
    actual trajectories here are navigation outcomes, not prediction targets.
    """
    scale = _condition_scale(condition)
    methods = []
    common_initial = None
    for method_id, predictor in zip(METHOD_IDS, _models(base, threshold, global_damping)):
        env = NavigationEnv(seed=int(seed), scenario=scenario, damping_scale=scale)
        state = env.reset(seed=int(seed))
        if common_initial is None:
            common_initial = state.copy()
        planner = MPCPlanner(predictor, horizon=10, adaptive=False, budget=4500,
                             seed=int(seed) + 90000)
        planner.reset()
        frames, actual, times = [], [state.copy()], []
        clearance, path_length = float('inf'), 0.0
        while not env.done:
            before = _status(predictor)
            started = time.perf_counter()
            action, plan = planner.act(state)
            planning_ms = (time.perf_counter() - started) * 1000
            next_state, _, _, _, final = env.step(action)
            started = time.perf_counter()
            planner.observe(state, action, next_state)
            update_ms = (time.perf_counter() - started) * 1000
            controller_ms = planning_ms + update_ms
            times.append(controller_ms)
            clearance = min(clearance, final['min_clearance'])
            path_length += float(np.linalg.norm(next_state[:2] - state[:2]))
            frames.append({
                'state': state.tolist(), 'next_state': next_state.tolist(),
                'action': action.tolist(), **plan, **before,
                'controller_ms': float(controller_ms), 'update_ms': float(update_ms),
                'gate_active_after': bool(getattr(predictor, 'gate_active', False)),
                'learned_scale_after': float(getattr(predictor, 'current_scale', 1)),
                'success': bool(final['success']), 'collision': bool(final['collision']),
                'min_clearance': float(final['min_clearance']),
            })
            actual.append(next_state.copy())
            state = next_state
        methods.append({
            'id': method_id, 'frames': frames, 'actual_path': np.asarray(actual).tolist(),
            'success': bool(final['success']), 'collision': bool(final['collision']),
            'timeout': bool(final['timeout']), 'steps': len(frames),
            'path_length': path_length, 'min_clearance': clearance,
            'mean_controller_ms': float(np.mean(times)), **_status(predictor),
        })
    return {
        'view': 'navigation', 'seed': int(seed), 'condition': condition,
        'scenario': scenario, 'initial_state': common_initial.tolist(), 'dt': DT,
        'methods': methods, 'field': _field_display(base, scale),
        'metadata': {
            'model_training_seed': 142, 'horizon': 10, 'member_transition_budget': 4500,
            'global_damping': float(global_damping),
            'controller_rng_seed': int(seed) + 90000,
            'candidate_count': '450 physics; 150 three-member learned (same4500 member-transition budget)',
            'online_update_budget': 'Observation updates additional to rollout budget; included in controller time',
            'metric': 'Paired navigation outcomes; actions diverge, so actual paths are not prediction-error targets',
            'field_access': 'Display/evaluator only; hidden truth field never enters a controller',
            'scope': 'One selectable diagnostic scene; aggregate claims require the frozen study',
        },
    }
