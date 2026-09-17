"""Transition collection with episode IDs for leakage-free dataset splitting."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import EnvConfig, NavigationEnv


def collect_dataset(episodes: int = 100, seed: int = 0,
                    path: str | Path | None = None, max_steps: int = 100,
                    scenarios: tuple[str, ...] = ("open", "crossing", "slalom"),
                    damping_scale: float = 1.0,
                    random_starts: bool = True) -> dict[str, np.ndarray]:
    """Collect transitions under diverse smooth, goal-biased exploratory actions.

    Returns ``states[N,18]``, ``actions[N,2]``, ``next_states[N,18]``, and
    ``episode_ids[N]``. ``episode_seeds[episodes]`` records initial conditions.
    Train/validation splits must use complete episode IDs, not random rows.
    ``seed`` controls both environment seeds and exploration. Optional ``path``
    writes a compressed NPZ containing the same arrays. ``random_starts`` adds
    safe distributed starts to two thirds of collection episodes for coverage;
    evaluation episodes retain the standard environment starts.
    """
    if episodes < 1 or max_steps < 1 or not scenarios:
        raise ValueError("episodes and max_steps must be positive; scenarios must be nonempty")
    rng = np.random.default_rng(seed)
    episode_seeds = rng.integers(0, 2 ** 31 - 1, size=episodes, dtype=np.int64)
    states, actions, next_states, episode_ids = [], [], [], []
    for episode in range(episodes):
        env = NavigationEnv(seed=int(episode_seeds[episode]),
                            scenario=scenarios[episode % len(scenarios)],
                            damping_scale=damping_scale,
                            config=EnvConfig(max_steps=max_steps))
        state = env.state
        if random_starts and episode % 3 != 0:
            obstacle_positions = np.stack([state[6 + i * 4:8 + i * 4] for i in range(3)])
            for _ in range(100):
                candidate = rng.uniform((0.75, 0.75), (9.25, 9.25))
                if np.min(np.linalg.norm(obstacle_positions - candidate, axis=1)) > 1.0:
                    state[:2] = candidate
                    break
            state[2:4] = rng.uniform(-0.5, 0.5, 2)
            for _ in range(100):
                goal = rng.uniform((0.8, 0.8), (9.2, 9.2))
                if np.linalg.norm(goal - state[:2]) > 2.5:
                    state[4:6] = goal
                    break
            env.state = state
        noise = rng.uniform(-1, 1, 2)
        # Non-goal waypoints explore vertical motion and varied field regions.
        waypoint = state[4:6].copy()
        exploratory_episode = episode % 3 == 0
        if exploratory_episode:
            waypoint = rng.uniform((1.0, 1.0), (9.0, 9.0))
        for t in range(max_steps):
            if exploratory_episode and (t % 18 == 0 or np.linalg.norm(waypoint - state[:2]) < 0.8):
                waypoint = rng.uniform((0.8, 0.8), (9.2, 9.2))
            target = waypoint - state[:2]
            target /= max(float(np.linalg.norm(target)), 0.1)
            noise = 0.72 * noise + 0.28 * rng.uniform(-1.0, 1.0, 2)
            if t % 12 == 0:
                noise = rng.uniform(-1.0, 1.0, 2)
            action = np.clip(0.65 * target - 0.20 * state[2:4] + 0.80 * noise, -1.0, 1.0)
            # Mild repulsion increases complete trajectories but preserves diverse near misses.
            for i in range(3):
                offset = state[:2] - state[6 + 4 * i:8 + 4 * i]
                distance = float(np.linalg.norm(offset))
                if 0.05 < distance < 1.3:
                    action += 0.45 * offset / distance * (1.3 - distance)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            next_state, _, terminated, truncated, _ = env.step(action)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            episode_ids.append(episode)
            state = next_state
            if terminated or truncated:
                break
    result = {"states": np.asarray(states, dtype=np.float32),
              "actions": np.asarray(actions, dtype=np.float32),
              "next_states": np.asarray(next_states, dtype=np.float32),
              "episode_ids": np.asarray(episode_ids, dtype=np.int64),
              "episode_seeds": episode_seeds}
    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **result)
    return result
