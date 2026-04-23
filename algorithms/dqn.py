"""
Independent DQN baseline for simple_world_comm.

Each agent has its own Q network and replay buffer.
No parameter sharing, no cross-agent communication.
This is the classic IL (Independent Learners) baseline.
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ═══════════════════════════════════════════════════════════
# Q Network
# ═══════════════════════════════════════════════════════════
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(act, dtype=torch.long),
            torch.tensor(rew, dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════
# Single-agent DQN
# ═══════════════════════════════════════════════════════════
class SingleDQN:
    def __init__(self, obs_dim, action_dim, cfg):
        self.device = torch.device(cfg.device)
        self.action_dim = action_dim

        self.q = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_q = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optim = optim.Adam(self.q.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.batch_size = 64
        self.target_update_every = 200
        self.step = 0

    def act(self, obs, eps):
        if random.random() < eps:
            return random.randint(0, self.action_dim - 1)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q(obs_t)
        return int(q.argmax(dim=-1).item())

    def update(self, buffer):
        if len(buffer) < self.batch_size:
            return {"q_loss": 0.0}

        obs, act, rew, next_obs, done = buffer.sample(self.batch_size)
        obs = obs.to(self.device)
        act = act.to(self.device)
        rew = rew.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        q_val = self.q(obs).gather(1, act.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_q(next_obs).max(1)[0]
            target = rew + self.gamma * (1 - done) * next_q

        loss = nn.MSELoss()(q_val, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optim.step()

        self.step += 1
        if self.step % self.target_update_every == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return {"q_loss": loss.item()}


# ═══════════════════════════════════════════════════════════
# Multi-agent wrapper (one DQN per agent)
# ═══════════════════════════════════════════════════════════
class DQNAgent:
    def __init__(self, cfg, env):
        self.env = env
        self.cfg = cfg

        self.agents: Dict[str, SingleDQN] = {}
        self.buffers: Dict[str, ReplayBuffer] = {}

        for name in env.possible_agents:
            spec = env.get_agent_spec(name)
            if spec.action_type != "discrete":
                raise ValueError(f"DQN requires discrete actions, got {spec.action_type} for {name}")
            self.agents[name] = SingleDQN(spec.obs_dim, spec.action_dim, cfg)
            self.buffers[name] = ReplayBuffer()

        # Role grouping for logging/eval
        self.adv_agents = [a for a in env.possible_agents
                           if "adversary" in a]
        self.good_agents = [a for a in env.possible_agents
                            if a.startswith("agent_")]

        # Epsilon schedule
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995

    # ── Required interface ──────────────────────────────
    def select_actions(self, obs, env=None, explore=True):
        eps = self.epsilon if explore else 0.0
        return {name: self.agents[name].act(ob, eps)
                for name, ob in obs.items()}

    def observe(self, transition):
        obs = transition["obs"]
        actions = transition["actions"]
        rewards = transition["rewards"]
        next_obs = transition["next_obs"]

        for a in obs.keys():
            if a not in self.agents:
                continue
            done = float(transition["terminated"].get(a, False)
                         or transition["truncated"].get(a, False))
            self.buffers[a].add(
                (obs[a], actions[a], rewards[a], next_obs[a], done)
            )

    def end_episode(self):
        # Decay epsilon once per episode instead of per update
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def update(self, global_step=None):
        logs = {}
        for name, agent in self.agents.items():
            log = agent.update(self.buffers[name])
            logs[f"{name}/q_loss"] = log["q_loss"]
        logs["epsilon"] = self.epsilon
        return logs

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(
            {k: v.q.state_dict() for k, v in self.agents.items()},
            os.path.join(path, "dqn.pt")
        )

    def load(self, path):
        fp = path if path.endswith(".pt") else os.path.join(path, "dqn.pt")
        state = torch.load(fp, map_location="cpu", weights_only=False)
        for k in self.agents:
            if k in state:
                self.agents[k].q.load_state_dict(state[k])
