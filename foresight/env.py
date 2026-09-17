"""Small, reproducible navigation world with hidden spatially varying damping.

Observation: [robot x,y,vx,vy, goal x,y, obstacle_0 x,y,vx,vy, ...].
All distances are metres and time is seconds.  Obstacles are exogenous; their
motion does not depend on robot actions.  No damping value enters observations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DT = 0.15
ROBOT_RADIUS = 0.18
OBSTACLE_RADIUS = 0.38
MAX_SPEED = 1.8
ACTION_GAIN = 3.0
STATE_DIM = 18
ACTION_DIM = 2


@dataclass(frozen=True)
class EnvConfig:
    size: float = 10.0
    dt: float = DT
    robot_radius: float = ROBOT_RADIUS
    obstacle_radius: float = OBSTACLE_RADIUS
    max_speed: float = MAX_SPEED
    action_gain: float = ACTION_GAIN
    goal_radius: float = 0.45
    max_steps: int = 100


def damping_at(positions: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Ground-truth damping; reserved for simulation/oracle baselines.

    A smooth but heterogeneous field makes prediction depend on position as
    well as velocity. ``scale`` changes the dynamics without changing the map.
    """
    pos = np.asarray(positions)
    x, y = pos[..., 0], pos[..., 1]
    field = (0.45 + 1.15 / (1.0 + np.exp(-1.4 * (x - 4.7)))
             + 0.40 * np.sin(0.75 * y) ** 2
             + 0.25 * np.sin(0.60 * x + 0.30 * y) ** 2)
    return np.asarray(field * scale, dtype=np.float32)


def _reflect(positions: np.ndarray, velocities: np.ndarray,
             dt: float, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact reflection, including travel through more than one wall."""
    length = high - low
    folded = np.mod(positions + velocities * dt - low, 2.0 * length)
    output_pos = low + np.where(folded <= length, folded, 2.0 * length - folded)
    output_vel = velocities * np.where(folded < length, 1.0, -1.0)
    output_vel = np.where(np.isclose(output_pos, low, rtol=0, atol=1e-7),
                          np.abs(output_vel), output_vel)
    output_vel = np.where(np.isclose(output_pos, high, rtol=0, atol=1e-7),
                          -np.abs(output_vel), output_vel)
    return output_pos, output_vel


def physics_step(states: np.ndarray, actions: np.ndarray, damping: np.ndarray | float,
                 *, config: EnvConfig | None = None) -> np.ndarray:
    """Vectorized one-step dynamics; broadcasting over any leading dimensions.

    ``damping`` broadcasts to the leading state/action shape (or may have a
    trailing singleton). This function neither checks nor terminates collisions.
    """
    cfg = config or EnvConfig()
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if states.shape[-1:] != (STATE_DIM,) or actions.shape[-1:] != (ACTION_DIM,):
        raise ValueError("Expected states[...,18] and actions[...,2].")
    lead = np.broadcast_shapes(states.shape[:-1], actions.shape[:-1])
    out = np.broadcast_to(states, lead + (STATE_DIM,)).copy()
    action = np.clip(np.broadcast_to(actions, lead + (ACTION_DIM,)), -1.0, 1.0)
    damping = np.asarray(damping, dtype=np.float32)
    if damping.ndim == len(lead) + 1 and damping.shape[-1] == 1:
        damping = np.squeeze(damping, axis=-1)
    damping = np.broadcast_to(damping, lead)
    velocity = (1.0 - damping[..., None] * cfg.dt) * out[..., 2:4]
    velocity += cfg.action_gain * action * cfg.dt
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    velocity *= np.minimum(1.0, cfg.max_speed / np.maximum(speed, 1e-8))
    raw_position = out[..., :2] + velocity * cfg.dt
    position = np.clip(raw_position, cfg.robot_radius, cfg.size - cfg.robot_radius)
    velocity = np.where(raw_position != position, 0.0, velocity)
    out[..., :2], out[..., 2:4] = position, velocity
    for i in range(3):
        start = 6 + 4 * i
        pos, vel = _reflect(out[..., start:start + 2], out[..., start + 2:start + 4],
                            cfg.dt, cfg.obstacle_radius, cfg.size - cfg.obstacle_radius)
        out[..., start:start + 2], out[..., start + 2:start + 4] = pos, vel
    return out


def _relative_segment_distance(a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    denominator = float(delta @ delta)
    t = float(np.clip(-(a @ delta) / denominator, 0, 1)) if denominator > 1e-15 else 0.0
    return float(np.linalg.norm(a + t * delta))


def _swept_obstacle_distance(robot_start: np.ndarray, robot_end: np.ndarray,
                             obstacle: np.ndarray, cfg: EnvConfig) -> float:
    """Minimum exact segment distance across every obstacle reflection event."""
    low, high = cfg.obstacle_radius, cfg.size - cfg.obstacle_radius
    pos, velocity = obstacle[:2].astype(float).copy(), obstacle[2:].astype(float).copy()
    elapsed, minimum = 0.0, float("inf")
    robot_velocity = (robot_end - robot_start) / cfg.dt
    # At a boundary, normalize velocity before computing the next event.
    for k in range(2):
        if pos[k] <= low + 1e-9 and velocity[k] < 0:
            velocity[k] *= -1
        if pos[k] >= high - 1e-9 and velocity[k] > 0:
            velocity[k] *= -1
    while elapsed < cfg.dt - 1e-12:
        hit_times = np.full(2, np.inf)
        for k in range(2):
            if velocity[k] > 0:
                hit_times[k] = (high - pos[k]) / velocity[k]
            elif velocity[k] < 0:
                hit_times[k] = (low - pos[k]) / velocity[k]
        remaining = cfg.dt - elapsed
        duration = min(remaining, max(0.0, float(np.min(hit_times))))
        end = pos + velocity * duration
        relative_start = robot_start + robot_velocity * elapsed - pos
        relative_end = robot_start + robot_velocity * (elapsed + duration) - end
        minimum = min(minimum, _relative_segment_distance(relative_start, relative_end))
        elapsed += duration
        pos = end
        for k in range(2):
            if hit_times[k] <= duration + 1e-10:
                velocity[k] *= -1
        if duration <= 1e-12 and not np.any(hit_times <= 1e-10):
            break
    return minimum


class NavigationEnv:
    """Navigation episodes with deterministic seeds and three moving obstacles."""

    def __init__(self, seed: int = 0, scenario: str = "crossing",
                 damping_scale: float = 1.0, config: EnvConfig | None = None):
        if scenario not in {"open", "crossing", "slalom"}:
            raise ValueError(f"Unknown scenario: {scenario}")
        if damping_scale <= 0:
            raise ValueError("damping_scale must be positive")
        self.config = config or EnvConfig()
        self.scenario = scenario
        self.damping_scale = float(damping_scale)
        self.rng = np.random.default_rng(seed)
        self._state = np.zeros(STATE_DIM, dtype=np.float32)
        self._trajectory: list[np.ndarray] = []
        self.steps = 0
        self.done = False
        self.reset()

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @state.setter
    def state(self, value: np.ndarray) -> None:
        state = np.asarray(value, dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite array of shape (18,)")
        self._state = state.copy()

    @property
    def trajectory(self) -> list[np.ndarray]:
        return [state.copy() for state in self._trajectory]

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg, rng = self.config, self.rng
        self._state[:] = 0
        y = rng.uniform(2.0, cfg.size - 2.0)
        self._state[:2] = (1.0, y)
        self._state[4:6] = (cfg.size - 1.0,
                             np.clip(y + rng.uniform(-1.7, 1.7), 1.0, cfg.size - 1.0))
        for i, x_fraction in enumerate((0.35, 0.53, 0.71)):
            start = 6 + 4 * i
            x = cfg.size * x_fraction + rng.uniform(-0.3, 0.3)
            if self.scenario == "open":
                obstacle_y = 0.8 if i % 2 == 0 else cfg.size - 0.8
                velocity = (rng.uniform(-0.20, 0.20), 0.0)
            elif self.scenario == "crossing":
                obstacle_y = np.clip(y + rng.uniform(-2.1, 2.1), 1.1, cfg.size - 1.1)
                velocity = (rng.uniform(-0.14, 0.14), rng.choice((-1.0, 1.0)) * rng.uniform(0.4, 0.9))
            else:
                obstacle_y = np.clip(y + (0.55 if i % 2 else -0.55) + rng.uniform(-0.2, 0.2),
                                     1.1, cfg.size - 1.1)
                velocity = (rng.uniform(-0.08, 0.08), rng.choice((-1.0, 1.0)) * rng.uniform(0.1, 0.3))
            self._state[start:start + 4] = (x, obstacle_y, *velocity)
        self.steps, self.done = 0, False
        self._trajectory = [self._state.copy()]
        return self.state

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("Episode ended; call reset() before step().")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite array of shape (2,)")
        old = self._state.copy()
        damping = float(damping_at(old[:2], self.damping_scale))
        self._state = physics_step(old, action, damping, config=self.config)
        cfg = self.config
        # Signed clearance accounts for walls as well as swept obstacle discs.
        endpoints = np.stack((old[:2], self._state[:2]))
        clearance = float(min(np.min(endpoints - cfg.robot_radius),
                              np.min(cfg.size - cfg.robot_radius - endpoints)))
        for i in range(3):
            obstacle = old[6 + i * 4:10 + i * 4]
            distance = _swept_obstacle_distance(old[:2], self._state[:2], obstacle, cfg)
            clearance = min(clearance, distance - cfg.robot_radius - cfg.obstacle_radius)
        collision = clearance <= 1e-6
        distance_before = float(np.linalg.norm(old[:2] - old[4:6]))
        distance_after = float(np.linalg.norm(self._state[:2] - self._state[4:6]))
        success = distance_after <= cfg.goal_radius and not collision
        self.steps += 1
        terminated = bool(collision or success)
        truncated = bool(self.steps >= cfg.max_steps and not terminated)
        self.done = terminated or truncated
        reward = distance_before - distance_after - 0.01 - 10.0 * collision + 10.0 * success
        self._trajectory.append(self._state.copy())
        info = {"success": bool(success), "collision": bool(collision), "timeout": truncated,
                "min_clearance": clearance, "damping": damping,
                "goal_distance": distance_after, "steps": self.steps}
        return self.state, float(reward), terminated, truncated, info
