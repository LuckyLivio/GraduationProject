"""Budgeted shooting MPC shared by learned and physical dynamics models.

The planner receives observations and a predictor, never an environment instance.
``model_steps`` counts every candidate transition for every model, including the
uncertainty probe. It is an inference budget, not a claim of equal wall time.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .env import physics_step


DT = 0.15
ACTION_GAIN = 3.0
MAX_SPEED = 1.8
ROBOT_RADIUS = 0.18
OBSTACLE_RADIUS = 0.38
WORLD_SIZE = 10.0


class PhysicsPredictor:
    """Nominal physics or a five-transition local least-squares identifier.

    The local identifier uses only past, unsaturated transitions. It does not
    inspect the hidden damping field. Wall/speed saturation is excluded because
    it would bias the damping estimate.
    """

    n_members = 1

    def __init__(self, damping: float = 0.8, adaptive: bool = False):
        self.initial_damping = float(damping)
        self.damping = float(damping)
        self.adaptive = bool(adaptive)
        self.history: deque[tuple[float, float]] = deque(maxlen=5)

    def reset(self):
        self.history.clear()
        self.damping = self.initial_damping

    def observe(self, state, action, next_state):
        if not self.adaptive:
            return
        s, a, ns = map(np.asarray, (state, action, next_state))
        velocity = s[2:4].astype(np.float64)
        next_velocity = ns[2:4].astype(np.float64)
        if np.linalg.norm(next_velocity) >= MAX_SPEED - 1e-4:
            return
        if np.any(ns[:2] <= ROBOT_RADIUS + 1e-4) or np.any(
            ns[:2] >= WORLD_SIZE - ROBOT_RADIUS - 1e-4
        ):
            return
        denominator = float(np.dot(velocity, velocity))
        if denominator < 1e-5:
            return
        residual = (velocity + ACTION_GAIN * np.clip(a, -1, 1) * DT - next_velocity) / DT
        self.history.append((float(np.dot(velocity, residual)), denominator))
        self.damping = float(np.clip(
            sum(x[0] for x in self.history) / sum(x[1] for x in self.history),
            0.0, 8.0,
        ))

    def rollout(self, state, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 3 or actions.shape[-1] != 2:
            raise ValueError("actions must have shape [candidates, horizon, 2]")
        n, horizon, _ = actions.shape
        paths = np.empty((1, n, horizon + 1, 18), dtype=np.float32)
        paths[0, :, 0] = state
        for t in range(horizon):
            paths[0, :, t + 1] = physics_step(paths[0, :, t], actions[:, t], self.damping)
        return paths


class MPCPlanner:
    """Random shooting with a common proposal/cost for all dynamics models.

    For adaptive planning, an independent small probe estimates the 90th
    percentile ensemble position disagreement at each future time. Both robot
    position and robot-obstacle relative position are checked. The longest
    configured horizon whose entire prefix is below the frozen threshold wins.
    The remaining budget is spent on control candidates at that horizon.
    """

    def __init__(
        self, predictor, horizon: int = 10, adaptive: bool = False,
        budget: int = 6000, seed: int = 0, uncertainty_threshold: float = 0.22,
        horizons=(5, 10, 16), probe_candidates: int = 8,
    ):
        self.predictor = predictor
        self.horizon = int(horizon)
        self.adaptive = bool(adaptive)
        self.budget = int(budget)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.horizons = tuple(sorted(set(map(int, horizons))))
        self.probe_candidates = int(probe_candidates)
        self.n_members = int(getattr(predictor, "n_members", 1))
        if not self.horizons or min(self.horizons) < 1 or self.horizon < 1:
            raise ValueError("horizons must be positive")
        if self.n_members < 1 or self.probe_candidates < 1:
            raise ValueError("member and probe counts must be positive")
        probe_cost = self.n_members * max(self.horizons) * self.probe_candidates if self.adaptive else 0
        required = probe_cost + self.n_members * (min(self.horizons) if self.adaptive else self.horizon)
        if self.budget < required:
            raise ValueError(f"budget must be at least {required} for the configured planner")
        self.rng = np.random.default_rng(seed)
        self.previous_actions = None

    def reset(self):
        self.previous_actions = None
        if hasattr(self.predictor, "reset"):
            self.predictor.reset()

    def observe(self, state, action, next_state):
        if hasattr(self.predictor, "observe"):
            self.predictor.observe(state, action, next_state)

    def _proposals(self, state, count, horizon):
        """Smooth controls around goal/left/right steering and previous plans.

        The steering proposal uses a fixed nominal integrator, identical for all
        methods; only predictor rollouts are used to select the action. Its
        nominal damping is not adapted to a test environment.
        """
        state = np.asarray(state, dtype=np.float32)
        delta = state[4:6] - state[:2]
        distance = float(np.linalg.norm(delta))
        forward = delta / max(distance, 1e-6)
        lateral = np.array([-forward[1], forward[0]], dtype=np.float32)
        offsets = np.array([0, -1.4, 1.4, -2.8, 2.8, -0.7, 0.7], dtype=np.float32)
        # Several side waypoints give shooting a useful route around obstacles.
        waypoints = np.broadcast_to(state[4:6], (len(offsets), 2)).copy()
        if distance > 1.0:
            waypoints[1:] = state[:2] + forward * min(2.8, distance) + offsets[1:, None] * lateral
        position = np.broadcast_to(state[:2], waypoints.shape).copy()
        velocity = np.broadcast_to(state[2:4], waypoints.shape).copy()
        references = np.empty((len(offsets), horizon, 2), dtype=np.float32)
        for t in range(horizon):
            error = waypoints - position
            norm = np.linalg.norm(error, axis=-1, keepdims=True)
            desired_velocity = error / np.maximum(norm, 1e-6) * np.minimum(1.65, norm * 1.8)
            control = np.clip((desired_velocity - velocity) * 2.4 / ACTION_GAIN + 0.8 * velocity / ACTION_GAIN, -1, 1)
            references[:, t] = control
            velocity = (1.0 - 0.8 * DT) * velocity + ACTION_GAIN * DT * control
            velocity *= np.minimum(1.0, MAX_SPEED / np.maximum(np.linalg.norm(velocity, axis=-1, keepdims=True), 1e-6))
            position += DT * velocity
        result = references[np.arange(count) % len(references)].copy()
        # Correlated noise explores curves, not only constant-angle paths.
        noise = self.rng.normal(size=(count, horizon, 2)).astype(np.float32)
        for t in range(1, horizon):
            noise[:, t] = 0.82 * noise[:, t - 1] + 0.57 * noise[:, t]
        scales = self.rng.uniform(0.15, 0.75, size=(count, 1, 1)).astype(np.float32)
        result += noise * scales
        exact = min(count, len(references))
        result[:exact] = references[:exact]
        if count >= 2:
            # This is explicit braking, not a zero-control coast.
            velocity = state[2:4].copy()
            for t in range(horizon):
                result[-1, t] = np.clip(-velocity / (ACTION_GAIN * DT), -1, 1)
                velocity = (1.0 - 0.8 * DT) * velocity + ACTION_GAIN * DT * result[-1, t]
        if self.previous_actions is not None and count >= 3:
            old = self.previous_actions
            tail = np.concatenate([old[1:], old[-1:]], axis=0)
            if len(tail) < horizon:
                tail = np.concatenate([tail, np.repeat(tail[-1:], horizon - len(tail), axis=0)])
            result[-2] = tail[:horizon]
        return np.clip(result, -1, 1).astype(np.float32)

    @staticmethod
    def disagreement(paths):
        """Disagreement in metres; mean-member RMS then candidate q90."""
        robot = paths[..., :2]
        robot_sigma = np.sqrt(np.mean(np.sum((robot - robot.mean(axis=0)) ** 2, axis=-1), axis=0))
        obstacles = paths[..., 6:].reshape(*paths.shape[:-1], 3, 4)[..., :2]
        relative = robot[..., None, :] - obstacles
        relative_sigma = np.sqrt(np.mean(np.sum((relative - relative.mean(axis=0)) ** 2, axis=-1), axis=0))
        combined = np.maximum(robot_sigma, relative_sigma.max(axis=-1))
        return np.quantile(combined, 0.90, axis=0)

    @staticmethod
    def _cost(paths, actions):
        robot = paths[..., :2]
        goal = paths[..., 4:6]
        distance = np.linalg.norm(robot - goal, axis=-1)
        obstacles = paths[..., 6:].reshape(*paths.shape[:-1], 3, 4)[..., :2]
        relative = robot[..., None, :] - obstacles
        start = relative[:, :, :-1]
        movement = relative[:, :, 1:] - start
        fraction = np.clip(-np.sum(start * movement, axis=-1) / np.maximum(np.sum(movement * movement, axis=-1), 1e-9), 0, 1)
        separation = np.linalg.norm(start + fraction[..., None] * movement, axis=-1)
        clearance = separation - (ROBOT_RADIUS + OBSTACLE_RADIUS)
        collision = np.any(clearance <= 0.025, axis=(-2, -1))
        boundary = np.min(np.minimum(robot[..., 1:, :], WORLD_SIZE - robot[..., 1:, :]), axis=(-2, -1))
        collision |= boundary <= ROBOT_RADIUS + 1e-4
        # Goal attraction, clearance, effort, and terminal speed all use the same
        # weights across fixed/adaptive and physical/learned methods.
        soft_risk = np.mean(np.exp(-np.maximum(clearance, -0.2) / 0.4), axis=(-2, -1))
        wall_risk = np.maximum(0, 0.55 - boundary) ** 2
        speed = np.linalg.norm(paths[..., -1, 2:4], axis=-1)
        stop_cost = speed ** 2 * np.exp(-distance[..., -1] / 0.6)
        member_cost = 9.0 * distance[..., -1] + 1.5 * distance[..., 1:].mean(axis=-1) + 7.0 * soft_risk + 12.0 * wall_risk + 2.0 * stop_cost
        member_cost += 0.08 * np.mean(actions ** 2, axis=(-2, -1))[None, :]
        score = member_cost.mean(axis=0) + 1000.0 * collision.mean(axis=0)
        return score, collision.any(axis=0)

    def act(self, state):
        started = time.perf_counter()
        state = np.asarray(state, dtype=np.float32)
        used = 0
        probe_disagreement = []
        horizon = self.horizon
        if self.adaptive:
            probe_horizon = max(self.horizons)
            probe_actions = self._proposals(state, self.probe_candidates, probe_horizon)
            probe_paths = self.predictor.rollout(state, probe_actions)
            used += self.n_members * self.probe_candidates * probe_horizon
            probe_sigma = self.disagreement(probe_paths)
            probe_disagreement = probe_sigma.tolist()
            affordable = [h for h in self.horizons if used + self.n_members * h <= self.budget]
            feasible = [h for h in affordable if float(np.max(probe_sigma[1:h + 1])) <= self.uncertainty_threshold]
            horizon = max(feasible) if feasible else min(affordable)
        count = (self.budget - used) // (self.n_members * horizon)
        actions = self._proposals(state, count, horizon)
        paths = self.predictor.rollout(state, actions)
        if paths.shape != (self.n_members, count, horizon + 1, 18):
            raise ValueError("predictor output shape does not match its n_members or rollout contract")
        used += self.n_members * count * horizon
        scores, colliding = self._cost(paths, actions)
        # If every sampled trajectory collides, use the explicit brake candidate.
        fallback = bool(np.all(colliding))
        choice = count - 1 if fallback and count >= 2 else int(np.argmin(scores))
        selected = actions[choice]
        self.previous_actions = selected.copy()
        order = np.argsort(scores)[:5]
        disagreement = self.disagreement(paths[:, choice:choice + 1])
        mean_paths = paths.mean(axis=0)
        info = {
            "horizon": int(horizon), "model_steps": int(used),
            "planning_ms": float((time.perf_counter() - started) * 1000),
            "n_candidates": int(count), "n_members": self.n_members,
            "disagreement": float(disagreement[-1]),
            "disagreement_by_step": disagreement.tolist(),
            "probe_disagreement": probe_disagreement,
            "uncertainty_threshold": self.uncertainty_threshold,
            "fallback": fallback, "predicted_path": mean_paths[choice].tolist(),
            "candidate_paths": mean_paths[order].tolist(),
            "selected_cost": float(scores[choice]),
            "predicted_collision": bool(colliding[choice]),
        }
        return selected[0].copy(), info
