from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os



# =========================
# Q Network
# =========================
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Replay Buffer
# =========================
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


# =========================
# Single DQN
# =========================
class SingleDQN:
    def __init__(self, obs_dim, action_dim, cfg):
        self.device = torch.device(cfg.device)

        self.q = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_q = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optim = optim.Adam(self.q.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.batch_size = 64
        self.tau_steps = 200
        self.step = 0

    def act(self, obs, eps):
        if random.random() < eps:
            return random.randint(0, self.q.net[-1].out_features - 1)

        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q(obs)
        return int(q.argmax(dim=-1).item())

    def update(self, buffer: ReplayBuffer):
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
        self.optim.step()

        self.step += 1
        if self.step % self.tau_steps == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return {"q_loss": loss.item()}


# =========================
# Multi-Agent DQN (FOR RUNTIME)
# =========================
class DQNMultiAgent:

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

        self.agents: Dict[str, SingleDQN] = {}
        self.buffers: Dict[str, ReplayBuffer] = {}

        for name in env.possible_agents:
            spec = env.get_agent_spec(name)

            self.agents[name] = SingleDQN(
                obs_dim=spec.obs_dim,
                action_dim=spec.action_dim,
                cfg=cfg
            )

            self.buffers[name] = ReplayBuffer()

        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        # =========================
# Role grouping (for runner compatibility)
# =========================
        self.adv_agents = [
            name for name in env.possible_agents
            if "adversary" in name
        ]

        self.good_agents = [
            name for name in env.possible_agents
            if name.startswith("agent_")
        ]

    # =========================
    # REQUIRED BY RUNNER
    # =========================
    def select_actions(self, obs, env=None, explore=True):
        actions = {}

        eps = self.epsilon if explore else 0.0

        for name, ob in obs.items():
            actions[name] = self.agents[name].act(ob, eps)

        return actions

    def observe(self, transition):
        obs = transition["obs"]
        actions = transition["actions"]
        rewards = transition["rewards"]
        next_obs = transition["next_obs"]

        dones = {
            a: float(transition["terminated"].get(a, False)
                     or transition["truncated"].get(a, False))
            for a in obs.keys()
        }

        for a in obs.keys():
            self.buffers[a].add(
                (obs[a], actions[a], rewards[a], next_obs[a], dones[a])
            )

    def end_episode(self):
        pass

    def update(self, global_step=None):
        logs = {}

        for name, agent in self.agents.items():
            log = agent.update(self.buffers[name])
            logs[f"{name}_q_loss"] = log["q_loss"]

        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
        logs["epsilon"] = self.epsilon

        return logs

    # =========================
    

    def save(self, path):
        os.makedirs(path, exist_ok=True) 

        torch.save(
            {k: v.q.state_dict() for k, v in self.agents.items()},
            os.path.join(path, "dqn.pt")
        )

    def load(self, path):
        state = torch.load(path)
        for k in self.agents:
            self.agents[k].q.load_state_dict(state[k])