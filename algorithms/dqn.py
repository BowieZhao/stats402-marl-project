"""
Stable DQN baseline (Double + Dueling + N-step + epsilon-greedy + role sharing).

Removed (vs the Rainbow-lite attempt):
  - NoisyNet  (caused training instability under multi-agent non-stationarity)
  - PER       (high-priority transitions go stale fast in non-stationary env)
  - Distributional / C51

Kept:
  - Double DQN (online picks, target evaluates)
  - Dueling network (V + A streams)
  - N-step returns (n=3)
  - Role parameter sharing (3 networks: leader, adversary, good)
  - Polyak soft target updates
  - Huber loss + grad clip
  - Epsilon-greedy with slower decay (more exploration than before)
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ═══════════════════════════════════════════════════════════
# Dueling Q network
# ═══════════════════════════════════════════════════════════
class DuelingQ(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.action_dim = action_dim
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, action_dim)

    def forward(self, x):
        z = self.feature(x)
        v = self.value(z)
        a = self.adv(z)
        return v + a - a.mean(dim=-1, keepdim=True)


# ═══════════════════════════════════════════════════════════
# N-step transition accumulator (per source agent)
# ═══════════════════════════════════════════════════════════
class NStepBuffer:
    def __init__(self, n=3, gamma=0.99):
        self.n = n
        self.gamma = gamma
        self.buf = deque(maxlen=n)

    def add(self, obs, act, rew, next_obs, done):
        self.buf.append((obs, act, rew, next_obs, done))
        if len(self.buf) < self.n and not done:
            return None
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
        out = []
        while len(self.buf) > 1:
            self.buf.popleft()
            if self.buf:
                out.append(self._build())
        self.buf.clear()
        return out


# ═══════════════════════════════════════════════════════════
# Replay buffer
# ═══════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buf = deque(maxlen=capacity)

    def add(self, transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


# ═══════════════════════════════════════════════════════════
# Per-role learner
# ═══════════════════════════════════════════════════════════
class RoleQLearner:
    def __init__(self, obs_dim, action_dim, cfg, name=""):
        self.device = torch.device(cfg.device)
        self.name = name
        self.action_dim = action_dim

        self.q = DuelingQ(obs_dim, action_dim).to(self.device)
        self.target = DuelingQ(obs_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=5e-4)

        self.gamma = 0.99
        self.n_step = 3
        self.batch_size = 128
        self.tau = 0.005
        self.warmup_steps = 2000

        self.buffer = ReplayBuffer(capacity=200000)
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

    def act(self, obs, eps):
        if random.random() < eps:
            return random.randint(0, self.action_dim - 1)
        obs_t = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q(obs_t)
        return int(q.argmax(dim=-1).item())

    def update(self):
        if len(self.buffer) < max(self.warmup_steps, self.batch_size):
            return 0.0

        batch = self.buffer.sample(self.batch_size)

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

        q_val = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)

        # Double DQN: online picks, target evaluates
        with torch.no_grad():
            next_actions = self.q(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            target = rew + (self.gamma ** n_steps) * (1.0 - done) * next_q

        loss = F.smooth_l1_loss(q_val, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

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

        self.learners: Dict[str, RoleQLearner] = {}
        if self.leader_names:
            spec = env.get_agent_spec(self.leader_names[0])
            self.learners["leader"] = RoleQLearner(spec.obs_dim, spec.action_dim, cfg, "leader")
        if self.normal_adv_names:
            spec = env.get_agent_spec(self.normal_adv_names[0])
            self.learners["adversary"] = RoleQLearner(spec.obs_dim, spec.action_dim, cfg, "adversary")
        if self.good_names:
            spec = env.get_agent_spec(self.good_names[0])
            self.learners["good"] = RoleQLearner(spec.obs_dim, spec.action_dim, cfg, "good")

        # Slower epsilon decay: reaches 0.1 around ep 1500, 0.05 by ep 3000
        # (was 0.995 -- too slow, good agents kept crashing boundaries)
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.998

    def _role(self, agent_name):
        if "leadadversary" in agent_name:
            return "leader"
        if "adversary" in agent_name:
            return "adversary"
        if agent_name.startswith("agent_"):
            return "good"
        return None

    def select_actions(self, obs, env=None, explore=True):
        eps = self.epsilon if explore else 0.0
        actions = {}
        for name, ob in obs.items():
            role = self._role(name)
            if role is None or role not in self.learners:
                continue
            actions[name] = self.learners[role].act(ob, eps)
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
        # Flush remaining n-step transitions at episode boundary
        for role, learner in self.learners.items():
            for name, nb in list(learner.n_step_buffers.items()):
                for t in nb.flush():
                    learner.buffer.add(t)
            learner.n_step_buffers.clear()
        # Decay epsilon
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

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
        logs["epsilon"] = self.epsilon
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
