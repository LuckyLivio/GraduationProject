"""Small learned, action-conditioned object dynamics with optional history.

Inputs contain observed object motion and commanded pushes only. Local-frame
coordinates encode rigid-motion invariance; contact dynamics are learned.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

HISTORY = 6


def current_features(states, actions):
    s = np.asarray(states, dtype=np.float32)
    a = np.asarray(actions, dtype=np.float32)
    c, sn = np.cos(s[..., 2]), np.sin(s[..., 2])
    velocity = np.stack((c*s[..., 3] + sn*s[..., 4],
                         -sn*s[..., 3] + c*s[..., 4], s[..., 5]), axis=-1)
    faces = np.eye(4, dtype=np.float32)[a[..., 0].astype(int)]
    return np.concatenate((velocity, faces, a[..., 1:]), axis=-1)


def transition_targets(states, next_states):
    s, n = np.asarray(states), np.asarray(next_states)
    c, sn = np.cos(s[..., 2]), np.sin(s[..., 2])
    d = n[..., :2] - s[..., :2]
    angle = (n[..., 2]-s[..., 2]+np.pi) % (2*np.pi)-np.pi
    return np.stack((c*d[..., 0]+sn*d[..., 1], -sn*d[..., 0]+c*d[..., 1], angle,
                     c*n[..., 3]+sn*n[..., 4], -sn*n[..., 3]+c*n[..., 4], n[..., 5]), axis=-1).astype(np.float32)


def apply_targets(states, targets):
    s, t = np.asarray(states), np.asarray(targets)
    c, sn = np.cos(s[..., 2]), np.sin(s[..., 2])
    return np.stack((s[..., 0]+c*t[..., 0]-sn*t[..., 1],
                     s[..., 1]+sn*t[..., 0]+c*t[..., 1],
                     (s[..., 2]+t[..., 2]+np.pi) % (2*np.pi)-np.pi,
                     c*t[..., 3]-sn*t[..., 4], sn*t[..., 3]+c*t[..., 4], t[..., 5]), axis=-1)


def training_arrays(states, actions):
    """Episode-boundary-safe examples. Padding is represented by a validity bit."""
    x = current_features(states[:, :-1], actions)
    y = transition_targets(states[:, :-1], states[:, 1:])
    records = np.concatenate((x, y, np.ones((*x.shape[:2], 1), dtype=np.float32)), axis=-1)
    history = np.zeros((*x.shape[:2], HISTORY, 16), dtype=np.float32)
    for t in range(x.shape[1]):
        count = min(t, HISTORY)
        if count:
            history[:, t, -count:] = records[:, t-count:t]
    return x.reshape(-1, 9), history.reshape(-1, HISTORY, 16), y.reshape(-1, 6)


class PushWorldModel(nn.Module):
    def __init__(self, use_history=True, width=128, context=64):
        super().__init__()
        self.use_history = use_history
        self.width, self.context = width, context
        self.memory = nn.GRU(16, context, batch_first=True) if use_history else None
        self.head = nn.Sequential(nn.Linear(9+(context if use_history else 0), width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 6))
        self.register_buffer('x_mean', torch.zeros(9))
        self.register_buffer('x_std', torch.ones(9))
        self.register_buffer('y_mean', torch.zeros(6))
        self.register_buffer('y_std', torch.ones(6))

    def set_normalization(self, x, y):
        for name, val in [('x_mean', x.mean(0)), ('x_std', np.maximum(x.std(0), .02)),
                          ('y_mean', y.mean(0)), ('y_std', np.maximum(y.std(0), .01))]:
            getattr(self, name).copy_(torch.as_tensor(val))

    def forward(self, x, history):
        xx = (x-self.x_mean)/self.x_std
        if self.use_history:
            valid = history[..., 15:16]
            hh = torch.cat(((history[..., :9]-self.x_mean)/self.x_std,
                            (history[..., 9:15]-self.y_mean)/self.y_std, valid), dim=-1)
            hh = hh * valid
            _, hidden = self.memory(hh)
            xx = torch.cat((xx, hidden[-1]), dim=-1)
        return self.head(xx)

    def history_array(self, states, actions):
        result = np.zeros((HISTORY, 16), dtype=np.float32)
        count = min(len(actions), HISTORY)
        if count:
            s, a = np.asarray(states), np.asarray(actions)
            x = current_features(s[-count-1:-1], a[-count:])
            y = transition_targets(s[-count-1:-1], s[-count:])
            result[-count:] = np.concatenate((x, y, np.ones((count, 1))), axis=-1)
        return result

    @torch.inference_mode()
    def predict_batch(self, states, actions, history):
        device = self.x_mean.device
        x = current_features(states, actions)
        h = np.asarray(history, dtype=np.float32)
        if h.ndim == 2:
            h = np.broadcast_to(h, (len(x), *h.shape)).copy()
        prediction = self(torch.as_tensor(x, device=device), torch.as_tensor(h, device=device))
        targets = (prediction*self.y_std+self.y_mean).cpu().numpy()
        return apply_targets(states, targets)

    def rollout(self, state, actions, history_states, history_actions):
        """Open loop: context is frozen at the last real observation.

        Imagined transitions never become new evidence about hidden physics.
        """
        context = self.history_array(history_states, history_actions)
        path = [np.asarray(state).copy()]
        for action in actions:
            path.append(self.predict_batch(np.asarray([path[-1]]), np.asarray([action]), context)[0])
        return np.asarray(path)

    def save(self, path, metadata):
        torch.save({'state_dict': self.cpu().state_dict(), 'use_history': self.use_history,
                    'width': self.width, 'context': self.context, 'metadata': metadata}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        if str(path).endswith('.npz'):
            with np.load(path, allow_pickle=False) as data:
                values = {k:torch.from_numpy(data[k].copy()) for k in data.files}
            use_history = 'memory.weight_ih_l0' in values
            width = values['head.0.weight'].shape[0]
            context = values['memory.weight_hh_l0'].shape[1] if use_history else 64
            model = cls(use_history, width, context)
            model.load_state_dict(values)
            return model.to(device).eval()
        saved = torch.load(path, map_location=device, weights_only=True)
        model = cls(saved['use_history'], saved['width'], saved['context'])
        model.load_state_dict(saved['state_dict'])
        return model.to(device).eval()
