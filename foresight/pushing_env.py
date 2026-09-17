"""Small planar rigid-body pushing task backed by real Pymunk contacts.

This is a task-level simulator, not an arm simulator: each macro action places a
kinematic circular pusher just outside an object face, moves it for 0.30 seconds,
then removes it and lets the object coast for 0.20 seconds. Repositioning is not
a collision-free path and must not be counted as a solved robot motion task.

Observations contain only geometric pose and twist. Hidden dynamics parameters
are supplied separately to the evaluator and are never returned in ``info``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pymunk


HALF_WIDTH = 0.25
HALF_HEIGHT = 0.18
PUSHER_RADIUS = 0.055
DT = 0.01
PUSH_STEPS = 30
COAST_STEPS = 20
MACRO_DURATION = (PUSH_STEPS + COAST_STEPS) * DT


@dataclass(frozen=True)
class PushParams:
    """Unobserved dynamics of one object, fixed throughout an episode.

    ``damping`` is the fraction of free velocity retained after one second
    (Pymunk's exponential damping), not a force coefficient. ``friction`` is
    contact friction against a pusher whose friction is 1.0. ``cog_x/y`` offset
    the center of gravity from the geometric center in the object's local frame.
    Mass is always 1 kg; this environment does not identify varying mass.
    """

    damping: float
    friction: float
    cog_x: float = 0.0
    cog_y: float = 0.0

    def __post_init__(self) -> None:
        values = (self.damping, self.friction, self.cog_x, self.cog_y)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Push parameters must be finite")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must lie in (0, 1]")
        if not 0.0 <= self.friction <= 2.0:
            raise ValueError("friction must lie in [0, 2]")
        if abs(self.cog_x) > 0.07 or abs(self.cog_y) > 0.07:
            raise ValueError("Each center-of-gravity offset must be at most 0.07 m")


def wrap_angle(angle: Any) -> Any:
    """Wrap radians to [-pi, pi), preserving arrays for vectorized metrics."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def sample_action(rng: np.random.Generator) -> np.ndarray:
    """Sample [face, offset, speed]; faces are left, right, bottom, top."""
    return np.array(
        [rng.integers(0, 4), rng.uniform(-0.75, 0.75), rng.uniform(0.15, 1.0)],
        dtype=np.float64,
    )


def sample_state(rng: np.random.Generator) -> np.ndarray:
    """Sample an object near the origin with a small initial geometric twist."""
    return np.array(
        [*rng.uniform(-0.2, 0.2, 2), rng.uniform(-np.pi, np.pi),
         *rng.uniform(-0.015, 0.015, 2), rng.uniform(-0.02, 0.02)],
        dtype=np.float64,
    )


def sample_params(rng: np.random.Generator) -> PushParams:
    """Training distribution; held-out studies should choose their own ranges."""
    return PushParams(
        damping=float(rng.uniform(0.05, 0.8)),
        friction=float(rng.uniform(0.1, 1.0)),
        cog_x=float(rng.uniform(-0.055, 0.055)),
        cog_y=float(rng.uniform(-0.055, 0.055)),
    )


def _state_of(body: pymunk.Body) -> np.ndarray:
    origin = body.local_to_world((0.0, 0.0))
    # Pymunk's velocity is the center-of-gravity velocity. Our observation is
    # the velocity at the geometric center, matching the reported position.
    velocity = body.velocity_at_local_point((0.0, 0.0))
    return np.array(
        [origin.x, origin.y, float(wrap_angle(body.angle)),
         velocity.x, velocity.y, body.angular_velocity],
        dtype=np.float64,
    )


def simulate_push(
    state: np.ndarray,
    action: np.ndarray,
    params: PushParams,
    record: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute one push and coast; no walls or hidden-state observations.

    Args:
        state: ``[geom_x, geom_y, angle, geom_vx, geom_vy, omega]``.
        action: ``[face, offset, speed]``. Offset is a fraction of the face's
            half-length. The push direction is fixed at action start.
        params: Evaluator-only object properties, constant within an episode.
        record: Include states and pusher positions every five physics steps.

    Returns:
        Independent next-state array and JSON-serializable diagnostics. Contact
        impulses are evaluator-only diagnostics, not model/planner observations.
        A fresh
        physics space is built per call so no contact cache or hidden state
        carries across macro actions. Inputs are never modified.
    """
    obs = np.asarray(state, dtype=np.float64)
    command = np.asarray(action, dtype=np.float64)
    if obs.shape != (6,) or not np.isfinite(obs).all():
        raise ValueError("state must be a finite array of shape (6,)")
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError("action must be a finite array of shape (3,)")
    face, offset, speed = command
    if face != int(face) or not 0 <= face <= 3:
        raise ValueError("face must be an integer in [0, 3]")
    if not -0.75 <= offset <= 0.75 or not 0.15 <= speed <= 1.0:
        raise ValueError("offset must be in [-0.75, 0.75], speed in [0.15, 1]")
    if not isinstance(params, PushParams):
        raise TypeError("params must be PushParams")

    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    space.damping = params.damping
    space.iterations = 30
    space.collision_slop = 0.0005
    # A fixed, explicitly defined inertia around the displaced center of
    # gravity. Offset objects are an abstract weighted box, not a uniform box.
    inertia = pymunk.moment_for_box(1.0, (2 * HALF_WIDTH, 2 * HALF_HEIGHT))
    inertia += params.cog_x ** 2 + params.cog_y ** 2
    body = pymunk.Body(1.0, inertia)
    body.center_of_gravity = (params.cog_x, params.cog_y)
    body.angle = float(obs[2])
    body.position = (float(obs[0]), float(obs[1]))
    # Explicitly enforce geometric origin even if body-position conventions
    # change across physics library versions.
    origin = body.local_to_world((0.0, 0.0))
    body.position += pymunk.Vec2d(float(obs[0] - origin.x), float(obs[1] - origin.y))
    body.angular_velocity = float(obs[5])
    cog_world = pymunk.Vec2d(params.cog_x, params.cog_y).rotated(float(obs[2]))
    body.velocity = (float(obs[3] - obs[5] * cog_world.y),
                     float(obs[4] + obs[5] * cog_world.x))
    box = pymunk.Poly.create_box(body, (2 * HALF_WIDTH, 2 * HALF_HEIGHT))
    box.friction = params.friction
    box.elasticity = 0.0
    box.collision_type = 1
    space.add(body, box)

    clearance = PUSHER_RADIUS + 0.004
    placements = (
        ((-HALF_WIDTH - clearance, offset * HALF_HEIGHT), (1.0, 0.0)),
        ((HALF_WIDTH + clearance, offset * HALF_HEIGHT), (-1.0, 0.0)),
        ((offset * HALF_WIDTH, -HALF_HEIGHT - clearance), (0.0, 1.0)),
        ((offset * HALF_WIDTH, HALF_HEIGHT + clearance), (0.0, -1.0)),
    )
    local_position, local_direction = placements[int(face)]
    pusher = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    pusher.position = body.local_to_world(local_position)
    direction = pymunk.Vec2d(*local_direction).rotated(float(obs[2]))
    pusher.velocity = direction * float(speed)
    tip = pymunk.Circle(pusher, PUSHER_RADIUS)
    tip.friction = 1.0
    tip.elasticity = 0.0
    tip.collision_type = 2
    space.add(pusher, tip)
    contact_steps = 0
    total_impulse = 0.0

    def on_contact(arbiter: pymunk.Arbiter, _space: pymunk.Space, _data: Any) -> None:
        nonlocal contact_steps, total_impulse
        contact_steps += 1
        total_impulse += arbiter.total_impulse.length

    handler = space.add_collision_handler(1, 2)
    handler.post_solve = on_contact
    frames: list[list[float]] = []
    positions: list[list[float] | None] = []
    times: list[float] = []

    def append_frame(t: float, pusher_present: bool) -> None:
        frames.append(_state_of(body).tolist())
        positions.append([float(pusher.position.x), float(pusher.position.y)]
                         if pusher_present else None)
        times.append(float(t))

    if record:
        append_frame(0.0, True)
    for step in range(PUSH_STEPS + COAST_STEPS):
        if step == PUSH_STEPS:
            space.remove(tip, pusher)
        space.step(DT)
        if record and (step + 1) % 5 == 0:
            append_frame((step + 1) * DT, step < PUSH_STEPS)
    info: dict[str, Any] = {
        "contact": bool(contact_steps),
        "contact_steps": int(contact_steps),
        "total_impulse": float(total_impulse),
        "duration": float(MACRO_DURATION),
    }
    if record:
        info.update(frames=frames, frame_times=times, pusher_positions=positions)
    return _state_of(body), info


def generate_dataset(
    episodes: int,
    steps: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Collect complete trajectories; split by episodes, never adjacent frames.

    Parameter labels are stored separately for evaluator stratification. They
    must not be concatenated into model observations or planner inputs.
    """
    if episodes < 1 or steps < 1:
        raise ValueError("episodes and steps must be positive")
    rng = np.random.default_rng(seed)
    states = np.empty((episodes, steps + 1, 6), dtype=np.float32)
    actions = np.empty((episodes, steps, 3), dtype=np.float32)
    contacts = np.empty((episodes, steps), dtype=np.bool_)
    hidden_params = np.empty((episodes, 4), dtype=np.float32)
    for episode in range(episodes):
        params = sample_params(rng)
        hidden_params[episode] = [params.damping, params.friction, params.cog_x, params.cog_y]
        state = sample_state(rng)
        states[episode, 0] = state
        for step in range(steps):
            action = sample_action(rng)
            state, info = simulate_push(state, action, params)
            states[episode, step + 1] = state
            actions[episode, step] = action
            contacts[episode, step] = info["contact"]
    return dict(states=states, actions=actions, contacts=contacts, hidden_params=hidden_params)
