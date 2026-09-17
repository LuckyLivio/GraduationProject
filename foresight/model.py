"""Hybrid learned world model: learned damping, known integration and geometry.

Training targets are inferred from observed transitions, never queried from the
environment's hidden damping field. This intentionally modest first model is
not a fully learned visual simulator. NumPy inference avoids a GPU sync at every
planning step; training uses PyTorch.
"""
from pathlib import Path
import time

import numpy as np

from .env import ACTION_GAIN, DT, MAX_SPEED, physics_step


def transition_targets(data):
    s, a, ns = data["states"], data["actions"], data["next_states"]
    v, vn = s[:, 2:4], ns[:, 2:4]
    energy = np.sum(v * v, axis=1)
    residual = v + ACTION_GAIN * a * DT - vn
    targets = np.sum(v * residual, axis=1) / (DT * np.maximum(energy, 1e-8))
    valid = (energy > .06) & (np.linalg.norm(vn, axis=1) < MAX_SPEED - 1e-3)
    valid &= np.all((ns[:, :2] > .1801) & (ns[:, :2] < 9.8199), axis=1)
    valid &= np.isfinite(targets) & (targets > 0) & (targets < 6)
    return s[valid, :2].astype(np.float32), targets[valid].astype(np.float32)


class HybridWorldModel:
    def __init__(self, members):
        self.members = members
        self.n_members = len(members)

    def damping(self, positions):
        # positions [M,N,2], member-specific weights
        output = []
        for m, weights in enumerate(self.members):
            x = positions[m] / 5.0 - 1.0
            for i in range(2):
                x = np.tanh(x @ weights[2 * i] + weights[2 * i + 1])
            z = x @ weights[4] + weights[5]
            output.append((.02 + 5.98 / (1 + np.exp(-np.clip(z[:, 0], -20, 20)))))
        return np.asarray(output, dtype=np.float32)

    def rollout(self, state, actions):
        actions = np.asarray(actions, dtype=np.float32)
        n, horizon = actions.shape[:2]
        states = np.broadcast_to(np.asarray(state, dtype=np.float32), (self.n_members, n, 18)).copy()
        result = np.empty((self.n_members, n, horizon + 1, 18), dtype=np.float32)
        result[:, :, 0] = states
        for k in range(horizon):
            states = physics_step(states, actions[None, :, k, :], self.damping(states[..., :2]))
            result[:, :, k + 1] = states
        return result

    def reset(self):
        pass

    def observe(self, state, action, next_state):
        # Frozen offline model; it does not adapt from evaluation transitions.
        pass

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f"m{m}_w{i}": w for m, ws in enumerate(self.members) for i, w in enumerate(ws)})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            count = len(data.files) // 6
            return cls([[data[f"m{m}_w{i}"].copy() for i in range(6)] for m in range(count)])


def train_world_model(train, validation, seed=42, epochs=80, members=3):
    import torch
    from torch import nn

    torch.set_num_threads(2)
    x, y = transition_targets(train)
    vx, vy = transition_targets(validation)
    if len(x) < 100 or len(vx) < 20:
        raise ValueError("Insufficient unsaturated, moving transitions to identify damping")
    start = time.perf_counter()
    tx = torch.from_numpy(x / 5 - 1)
    ty = torch.from_numpy(y[:, None])
    tvx = torch.from_numpy(vx / 5 - 1)
    tvy = torch.from_numpy(vy[:, None])
    ensemble, histories = [], []
    for member in range(members):
        torch.manual_seed(seed + member * 997)
        rng = np.random.default_rng(seed + member * 997)
        net = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        optimizer = torch.optim.Adam(net.parameters(), lr=.003)
        # Bootstrap complete episodes first, then use valid transitions belonging
        # to sampled episodes. Mapping through the same target mask preserves groups.
        all_s, all_ns = train['states'], train['next_states']
        velocity = all_s[:, 2:4]
        energy = (velocity * velocity).sum(1)
        raw = (velocity * (velocity + ACTION_GAIN * train['actions'] * DT - all_ns[:, 2:4])).sum(1) / (DT * np.maximum(energy, 1e-8))
        valid = (energy > .06) & (np.linalg.norm(all_ns[:, 2:4], axis=1) < MAX_SPEED - 1e-3) & np.isfinite(raw) & (raw > 0) & (raw < 6)
        valid &= np.all((all_ns[:, :2] > .1801) & (all_ns[:, :2] < 9.8199), axis=1)
        ids = train['episode_ids'][valid]
        unique = np.unique(ids)
        sampled = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([np.flatnonzero(ids == ep) for ep in sampled])
        best_loss, best = float('inf'), None
        history = []
        for epoch in range(epochs):
            net.train()
            shuffled = rng.permutation(index)
            for offset in range(0, len(shuffled), 512):
                indices = shuffled[offset:offset + 512]
                prediction = .02 + 5.98 * torch.sigmoid(net(tx[indices]))
                loss = (prediction - ty[indices]).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                val_loss = float((.02 + 5.98 * torch.sigmoid(net(tvx)) - tvy).square().mean())
            if val_loss < best_loss:
                best_loss = val_loss
                best = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if epoch % 10 == 0 or epoch == epochs - 1:
                history.append({'epoch': epoch + 1, 'validation_damping_mse': val_loss})
        net.load_state_dict(best)
        weights = []
        for layer in [net[0], net[2], net[4]]:
            weights.extend([layer.weight.detach().numpy().T.copy(), layer.bias.detach().numpy().copy()])
        ensemble.append(weights)
        histories.append(history)
        print(f'member {member + 1}/{members}: held-out damping RMSE={best_loss ** .5:.4f}', flush=True)
    return HybridWorldModel(ensemble), {
        'training_seed': seed, 'ensemble_members': members, 'independent_training_runs': 1,
        'architecture': '3 x MLP(2,32,32,1); learned position-dependent damping; known action integration and obstacle motion',
        'epochs_per_member': epochs, 'valid_training_transitions': len(x), 'valid_validation_transitions': len(vx),
        'training_seconds': time.perf_counter() - start, 'device': 'cpu', 'history': histories,
        'global_damping_fit': float(np.mean(y)),
    }
