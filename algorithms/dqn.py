"""
Rainbow-lite DQN baseline (Double + Dueling + PER + NoisyNet + N-step).

Skipped components vs full Rainbow:
  - Distributional / C51 (large code change, marginal gains here)

Design (matches MAPPO's RoleGroup structure):
  - 3 networks: leader, normal_adversary (shared across 3 adv), good
  - Each role has its own PER buffer + N-step accumulator
  - Noisy linear layers replace epsilon-greedy exploration
"""

from __future__ import annotations

import math
import os
import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ═══════════════════════════════════════════════════════════
# NoisyLinear (replaces epsilon-greedy)
# ═══════════════════════════════════════════════════════════
class NoisyLinear(nn.Module):
    """Factorised Gaussian noise on weights — Fortunato et al. 2017."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# ═══════════════════════════════════════════════════════════
# Dueling Q network with NoisyLinear
# ═══════════════════════════════════════════════════════════
class DuelingNoisyQ(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.action_dim = action_dim

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )
        # Dueling: value stream + advantage stream
        self.value_hidden = NoisyLinear(hidden, hidden)
        self.value_out = NoisyLinear(hidden, 1)

        self.adv_hidden = NoisyLinear(hidden, hidden)
        self.adv_out = NoisyLinear(hidden, action_dim)

    def forward(self, x):
        z = self.feature(x)
        v = self.value_out(F.relu(self.value_hidden(z)))         # (B, 1)
        a = self.adv_out(F.relu(self.adv_hidden(z)))             # (B, A)
        # Q = V + A - mean(A)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self):
        for m in [self.value_hidden, self.value_out,
                  self.adv_hidden, self.adv_out]:
            m.reset_noise()


# ═══════════════════════════════════════════════════════════
# Prioritized Experience Replay (PER)
# ═══════════════════════════════════════════════════════════
class SumTree:
    """Binary tree where parent = sum of children. O(log n) sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.max_priority = 1.0

    def add(self, transition):
        # New transitions get max priority so they're sampled at least once
        self.tree.add(self.max_priority ** self.alpha, transition)

    def __len__(self):
        return self.tree.size

    def sample(self, batch_size):
        beta = min(1.0,
                   self.beta_start + (1.0 - self.beta_start) *
                   self.frame / self.beta_frames)
        self.frame += 1

        batch, idxs, priorities = [], [], []
        seg = self.tree.total() / batch_size
        for i in range(batch_size):
            s = random.uniform(seg * i, seg * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None:
                # Fallback if sumtree returns empty; resample uniformly
                while data is None:
                    s = random.uniform(0, self.tree.total())
                    idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        priorities = np.array(priorities, dtype=np.float64)
        probs = priorities / self.tree.total()
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()

        return batch, idxs, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, idxs, priorities):
        for idx, p in zip(idxs, priorities):
            p = float(p)
            self.max_priority = max(self.max_priority, p)
            self.tree.update(idx, (p + 1e-6) ** self.alpha)


# ═══════════════════════════════════════════════════════════
# N-step transition accumulator
# ═══════════════════════════════════════════════════════════
class NStepBuffer:
    """Accumulates N consecutive 1-step transitions into a single
    N-step transition for bootstrapped TD targets."""
    def __init__(self, n=3, gamma=0.99):
        self.n = n
        self.gamma = gamma
        self.buf = deque(maxlen=n)

    def add(self, obs, act, rew, next_obs, done):
        self.buf.append((obs, act, rew, next_obs, done))
        if len(self.buf) < self.n and not done:
            return None  # not yet full
        return self._build()

    def _build(self):
        obs, act = self.buf[0][0], self.buf[0][1]
        cum_rew = 0.0
        next_obs, done = None, False
        for i, (_, _, r, nob, d) in enumerate(self.buf):
            cum_rew += (self.gamma ** i) * r
            next_obs = nob
            if d:
                done = True
                break
        return (obs, act, cum_rew, next_obs, done, len(self.buf))

    def flush(self):
        """Drain remaining transitions at episode end."""
        out = []
        while len(self.buf) > 1:
            self.buf.popleft()
            if self.buf:
                out.append(self._build())
        self.buf.clear()
        return out


# ═══════════════════════════════════════════════════════════
# Per-role learner
# ═══════════════════════════════════════════════════════════
class RoleRainbowLearner:
    def __init__(self, obs_dim, action_dim, cfg, name=""):
        self.device = torch.device(cfg.device)
        self.name = name
        self.action_dim = action_dim

        self.q = DuelingNoisyQ(obs_dim, action_dim).to(self.device)
        self.target = DuelingNoisyQ(obs_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=2.5e-4, eps=1e-5)

        self.gamma = 0.99
        self.n_step = 3
        self.batch_size = 128
        self.tau = 0.005   # polyak (NoisyNet handles exploration; soft target)
        self.warmup_steps = 2000

        self.buffer = PrioritizedReplayBuffer(capacity=200000,
                                              alpha=0.6,
                                              beta_start=0.4,
                                              beta_frames=200000)

        # One n-step accumulator per source agent. We use a dict keyed
        # by agent_name (passed in via add_transition) since multiple agents
        # share this learner.
        self.n_step_buffers: Dict[str, NStepBuffer] = {}

    def add_transition(self, agent_name, obs, act, rew, next_obs, done):
        if agent_name not in self.n_step_buffers:
            self.n_step_buffers[agent_name] = NStepBuffer(self.n_step, self.gamma)
        nb = self.n_step_buffers[agent_name]

        n_trans = nb.add(obs, act, rew, next_obs, done)
        if n_trans is not None:
            self.buffer.add(n_trans)
        if done:
            for t in nb.flush():
                self.buffer.add(t)
            del self.n_step_buffers[agent_name]

    def act(self, obs):
        # NoisyNet IS the exploration mechanism; just sample noise and argmax
        self.q.train()  # need train mode for noise to apply
        self.q.reset_noise()
        obs_t = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q(obs_t)
        return int(q.argmax(dim=-1).item())

    def act_eval(self, obs):
        self.q.eval()
        obs_t = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q(obs_t)
        return int(q.argmax(dim=-1).item())

    def update(self):
        if len(self.buffer) < max(self.warmup_steps, self.batch_size):
            return 0.0

        batch, idxs, weights = self.buffer.sample(self.batch_size)
        weights = weights.to(self.device)

        obs = torch.tensor(np.array([t[0] for t in batch]),
                           dtype=torch.float32, device=self.device)
        act = torch.tensor([t[1] for t in batch],
                           dtype=torch.long, device=self.device)
        rew = torch.tensor([t[2] for t in batch],
                           dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(np.array([t[3] for t in batch]),
                                dtype=torch.float32, device=self.device)
        done = torch.tensor([float(t[4]) for t in batch],
                            dtype=torch.float32, device=self.device)
        n_steps = torch.tensor([t[5] for t in batch],
                               dtype=torch.float32, device=self.device)

        # Reset noise for online & target nets at start of each update
        self.q.reset_noise()
        self.target.reset_noise()

        # Online Q value
        q_val = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)

        # Double DQN: online picks action, target evaluates
        with torch.no_grad():
            self.q.reset_noise()
            next_actions = self.q(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            # N-step bootstrap: gamma^n
            target = rew + (self.gamma ** n_steps) * (1.0 - done) * next_q

        td_errors = q_val - target
        # Importance-sampling-weighted Huber loss
        loss = (weights * F.smooth_l1_loss(q_val, target, reduction="none")).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

        # Update PER priorities
        new_priorities = td_errors.detach().abs().cpu().numpy() + 1e-6
        self.buffer.update_priorities(idxs, new_priorities)

        # Polyak target update
        for p, tp in zip(self.q.parameters(), self.target.parameters()):
            tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

        return float(loss.item())


# ═══════════════════════════════════════════════════════════
# Multi-agent wrapper
# ═══════════════════════════════════════════════════════════
class DQNAgent:
    def __init__(self, cfg, env):
        self.env = env
        self.cfg = cfg

        self.leader_names = [a for a in env.possible_agents if "leadadversary" in a]
        self.normal_adv_names = [a for a in env.possible_agents
                                  if "adversary" in a and "leadadversary" not in a]
        self.good_names = [a for a in env.possible_agents if a.startswith("agent_")]

        self.adv_agents = self.leader_names + self.normal_adv_names
        self.good_agents = self.good_names

        self.learners: Dict[str, RoleRainbowLearner] = {}
        if self.leader_names:
            spec = env.get_agent_spec(self.leader_names[0])
            self.learners["leader"] = RoleRainbowLearner(spec.obs_dim, spec.action_dim, cfg, "leader")
        if self.normal_adv_names:
            spec = env.get_agent_spec(self.normal_adv_names[0])
            self.learners["adversary"] = RoleRainbowLearner(spec.obs_dim, spec.action_dim, cfg, "adversary")
        if self.good_names:
            spec = env.get_agent_spec(self.good_names[0])
            self.learners["good"] = RoleRainbowLearner(spec.obs_dim, spec.action_dim, cfg, "good")

    def _role(self, agent_name):
        if "leadadversary" in agent_name:
            return "leader"
        if "adversary" in agent_name:
            return "adversary"
        if agent_name.startswith("agent_"):
            return "good"
        return None

    def select_actions(self, obs, env=None, explore=True):
        actions = {}
        for name, ob in obs.items():
            role = self._role(name)
            if role is None or role not in self.learners:
                continue
            if explore:
                actions[name] = self.learners[role].act(ob)
            else:
                actions[name] = self.learners[role].act_eval(ob)
        return actions

    def observe(self, transition):
        obs = transition["obs"]
        actions = transition["actions"]
        rewards = transition["rewards"]
        next_obs = transition["next_obs"]

        for a in obs:
            role = self._role(a)
            if role is None or role not in self.learners:
                continue
            done = float(transition["terminated"].get(a, False)
                         or transition["truncated"].get(a, False))
            self.learners[role].add_transition(
                a,
                np.asarray(obs[a], dtype=np.float32),
                int(actions[a]),
                float(rewards[a]),
                np.asarray(next_obs[a], dtype=np.float32),
                done
            )

    def end_episode(self):
        # Flush any remaining n-step transitions at ep boundary
        for role, learner in self.learners.items():
            for name, nb in list(learner.n_step_buffers.items()):
                for t in nb.flush():
                    learner.buffer.add(t)
            learner.n_step_buffers.clear()

    def update(self, global_step=None):
        logs = {}
        n_updates = 4
        for role, learner in self.learners.items():
            losses = []
            for _ in range(n_updates):
                l = learner.update()
                if l > 0:
                    losses.append(l)
            if losses:
                logs[f"{role}/q_loss"] = float(np.mean(losses))
        return logs

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        state = {role: learner.q.state_dict()
                 for role, learner in self.learners.items()}
        torch.save(state, os.path.join(path, "dqn.pt"))

    def load(self, path):
        fp = path if path.endswith(".pt") else os.path.join(path, "dqn.pt")
        state = torch.load(fp, map_location="cpu", weights_only=False)
        for role, sd in state.items():
            if role in self.learners:
                self.learners[role].q.load_state_dict(sd)
