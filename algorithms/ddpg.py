from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# Actor
# =========================
class Actor(nn.Module):
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
# Critic
# =========================
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
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
            torch.tensor(np.array(act), dtype=torch.float32),
            torch.tensor(np.array(rew), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)


# =========================
# Single DDPG Agent
# =========================
class SingleDDPG:
    def __init__(self, obs_dim, action_dim, cfg):
        self.device = torch.device(cfg.device)

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.tau = 0.01
        self.batch_size = 64

    # discrete action (soft DDPG hack)
    def act(self, obs, noise=True):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.actor(obs)
            action = torch.argmax(logits, dim=-1).item()

        if noise and random.random() < 0.1:
            action = random.randint(0, logits.shape[-1] - 1)

        return action

    def soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def update(self, buffer: ReplayBuffer):

        if len(buffer) < self.batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "q_loss": 0.0
            }

        obs, act, rew, next_obs, done = buffer.sample(self.batch_size)

        obs = obs.to(self.device)
        act = act.to(self.device)
        rew = rew.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        act_onehot = torch.nn.functional.one_hot(
            act.squeeze().long(),
            num_classes=self.actor.net[-1].out_features
        ).float()

        # =========================
        # Critic loss
        # =========================
        q = self.critic(obs, act_onehot)

        with torch.no_grad():
            next_logits = self.actor_target(next_obs)
            next_probs = torch.softmax(next_logits, dim=-1)

            target_q = self.critic_target(next_obs, next_probs)
            y = rew + self.gamma * (1 - done) * target_q

        critic_loss = nn.MSELoss()(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # =========================
        # Actor loss (stable version)
        # =========================
        logits = self.actor(obs)
        probs = torch.softmax(logits, dim=-1)

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        actor_loss = -self.critic(obs, probs).mean() - 0.01 * entropy

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # soft update
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "q_loss": float(critic_loss.item())
        }


# =========================
# Multi-Agent Wrapper
# =========================
class DDPGMultiAgent:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

        self.agents: Dict[str, SingleDDPG] = {}
        self.buffers: Dict[str, ReplayBuffer] = {}

        for name in env.possible_agents:
            spec = env.get_agent_spec(name)

            self.agents[name] = SingleDDPG(
                obs_dim=spec.obs_dim,
                action_dim=spec.action_dim,
                cfg=cfg
            )
            self.buffers[name] = ReplayBuffer()

    # =========================
    # REQUIRED BY EXPERIMENT
    # =========================
    @property
    def adv_agents(self):
        return [a for a in self.agents if "adversary" in a or "lead" in a]

    @property
    def good_agents(self):
        return [a for a in self.agents if "agent" in a]

    # =========================
    def select_actions(self, obs, env=None, explore=True):
        return {
            a: self.agents[a].act(o, noise=explore)
            for a, o in obs.items()
        }

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

    def update(self, global_step=None):
        logs = {}

        for name, agent in self.agents.items():
            out = agent.update(self.buffers[name])
            logs[f"{name}_actor_loss"] = out["actor_loss"]
            logs[f"{name}_critic_loss"] = out["critic_loss"]
            logs[f"{name}_q_loss"] = out["q_loss"]

        return logs

    def end_episode(self):
        pass

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(
            {k: v.actor.state_dict() for k, v in self.agents.items()},
            os.path.join(path, "ddpg_actor.pt")
        )

    def load(self, path):
        state = torch.load(path)
        for k in self.agents:
            self.agents[k].actor.load_state_dict(state[k])