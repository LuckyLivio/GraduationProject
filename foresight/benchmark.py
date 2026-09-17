"""Held-out gate-calibration scenarios; true changes remain environment-only."""
import numpy as np

from .env import NavigationEnv

CONDITIONS = ('nominal', 'global_shift', 'local_patch', 'switch_recover', 'sensor_noise')
SHIFT_STEP = 8
RECOVER_STEP = 24


class BenchmarkWorld(NavigationEnv):
    def __init__(self, seed, scenario, condition):
        if condition not in CONDITIONS:
            raise ValueError(f'Unknown benchmark condition: {condition}')
        self.condition = condition
        super().__init__(seed=seed, scenario=scenario)

    def step(self, action):
        if self.condition == 'global_shift':
            self.damping_scale = 1.7
        elif self.condition == 'local_patch':
            # Smooth local spatial change, not a global multiplier. The model's
            # multiplicative correction is deliberately misspecified here.
            x, y = self.state[:2]
            self.damping_scale = 1 + 1.2 * np.exp(-((x - 5.2) / 1.3) ** 2 - ((y - 5) / 3) ** 2)
        elif self.condition == 'switch_recover':
            self.damping_scale = 1.7 if SHIFT_STEP <= self.steps < RECOVER_STEP else 1.
        else:
            self.damping_scale = 1.
        state, reward, terminated, truncated, info = super().step(action)
        info['applied_scale'] = float(self.damping_scale)  # evaluator only
        return state, reward, terminated, truncated, info


def sensed_state(state, condition, seed, step):
    """Common exogenous sensor noise by seed and timestep, not action history.

    Only robot position and velocity are noisy. Goal/obstacle observations stay
    exact. Controllers receive this same convention; simulator scoring uses truth.
    """
    observed = np.asarray(state, dtype=np.float32).copy()
    if condition == 'sensor_noise':
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 903]))
        observed[:2] += rng.normal(0, .005, 2)
        observed[2:4] += rng.normal(0, .02, 2)
    return observed


def collect_prediction_episodes(condition, seed=75000, episodes=24):
    """Independent goal-biased behavior traces for same-action open-loop error."""
    traces = []
    for i in range(episodes):
        episode_seed = seed + i
        env = BenchmarkWorld(episode_seed, ('open', 'crossing', 'slalom')[i % 3], condition)
        truth = env.reset(seed=episode_seed)
        rng = np.random.default_rng(episode_seed + 100000)
        action_noise = np.zeros(2)
        states = [truth.copy()]
        observations = [sensed_state(truth, condition, episode_seed, 0)]
        actions = []
        scales = []
        for step in range(100):
            target = truth[4:6] - truth[:2]
            target /= max(float(np.linalg.norm(target)), .1)
            action_noise = .75 * action_noise + .25 * rng.uniform(-1, 1, 2)
            action = np.clip(.65 * target - .20 * truth[2:4] + .8 * action_noise, -1, 1)
            truth, _, ended, timed_out, info = env.step(action)
            states.append(truth.copy())
            observations.append(sensed_state(truth, condition, episode_seed, step + 1))
            actions.append(action.copy())
            scales.append(info['applied_scale'])
            if ended or timed_out:
                break
        traces.append({'seed': episode_seed, 'states': np.asarray(states),
                       'observations': np.asarray(observations), 'actions': np.asarray(actions),
                       'scales': np.asarray(scales)})
    return traces
