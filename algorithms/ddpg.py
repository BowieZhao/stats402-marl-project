"""
Independent DDPG baseline (discrete actions via Gumbel-Softmax).

DDPG is natively for continuous actions. To use it here with discrete actions,
the actor outputs logits, and we sample actions via Gumbel-Softmax so that the
critic receives a consistent action representation in both training phases.

This is a pragmatic adaptation — DDPG is expected to underperform MAPPO/DQN
on discrete multi-agent tasks, which is useful as a baseline.
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
# Gumbel-Softmax helper
# ═══════════════════════════════════════════════════════════
def gumbel_softmax(logits, tau=1.0, hard=False):
    """
    Differentiable discrete sampling.
    If hard=True, returns one-hot in forward but soft gradient in backward.
    """
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = F.softmax((logits + g) / tau, dim=-1)
    if hard:
        idx = y.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
        y = (y_hard - y).detach() + y
    return y


# ═══════════════════════════════════════════════════════════
# Actor and Critic
# ═══════════════════════════════════════════════════════════
class Actor(nn.Module):
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
        return self.net(x)  # logits


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
        return self.net(torch.cat([obs, act], dim=-1))


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
            torch.tensor(rew, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════
# Single DDPG agent
# ═══════════════════════════════════════════════════════════
class SingleDDPG:
    def __init__(self, obs_dim, action_dim, cfg):
        self.device = torch.device(cfg.device)
        self.action_dim = action_dim

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.tau = 0.01
        self.batch_size = 64

        # Exploration schedule (like DQN, simpler and more honest than fixed noise)
        self.epsilon = 1.0
        self.eps_min = 0.05

    def act(self, obs, explore=True):
        # Epsilon-greedy on argmax for discrete setting
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(obs_t)
        return int(logits.argmax(dim=-1).item())

    def soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def update(self, buffer):
        if len(buffer) < self.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        obs, act, rew, next_obs, done = buffer.sample(self.batch_size)
        obs = obs.to(self.device)
        act = act.to(self.device)
        rew = rew.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        act_onehot = F.one_hot(act, num_classes=self.action_dim).float()

        # ── Critic update ─────────────────────────────────
        q = self.critic(obs, act_onehot)
        with torch.no_grad():
            next_logits = self.actor_target(next_obs)
            # Use Gumbel-Softmax hard sampling for consistent one-hot input
            next_act = gumbel_softmax(next_logits, tau=1.0, hard=True)
            target_q = self.critic_target(next_obs, next_act)
            y = rew + self.gamma * (1.0 - done) * target_q

        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor update ──────────────────────────────────
        # Actor also uses Gumbel-Softmax hard sample so critic sees one-hot.
        logits = self.actor(obs)
        act_soft = gumbel_softmax(logits, tau=1.0, hard=True)
        actor_loss = -self.critic(obs, act_soft).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Soft target updates ───────────────────────────
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
        }


# ═══════════════════════════════════════════════════════════
# Multi-agent wrapper
# ═══════════════════════════════════════════════════════════
class DDPGAgent:
    def __init__(self, cfg, env):
        self.env = env
        self.cfg = cfg

        self.agents: Dict[str, SingleDDPG] = {}
        self.buffers: Dict[str, ReplayBuffer] = {}

        for name in env.possible_agents:
            spec = env.get_agent_spec(name)
            if spec.action_type != "discrete":
                raise ValueError(f"This DDPG adaptation requires discrete actions, got {spec.action_type}")
            self.agents[name] = SingleDDPG(spec.obs_dim, spec.action_dim, cfg)
            self.buffers[name] = ReplayBuffer()

        self.adv_agents = [a for a in env.possible_agents if "adversary" in a]
        self.good_agents = [a for a in env.possible_agents if a.startswith("agent_")]

        self.eps_decay = 0.995

    # ── Required interface ──────────────────────────────
    def select_actions(self, obs, env=None, explore=True):
        return {name: self.agents[name].act(ob, explore=explore)
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
        # Decay epsilon per episode for all agents
        for agent in self.agents.values():
            agent.epsilon = max(agent.epsilon * self.eps_decay, agent.eps_min)

    def update(self, global_step=None):
        logs = {}
        for name, agent in self.agents.items():
            log = agent.update(self.buffers[name])
            logs[f"{name}/actor_loss"] = log["actor_loss"]
            logs[f"{name}/critic_loss"] = log["critic_loss"]
        # Log epsilon of any one agent (they all decay together)
        if self.agents:
            logs["epsilon"] = next(iter(self.agents.values())).epsilon
        return logs

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(
            {k: {"actor": v.actor.state_dict(),
                 "critic": v.critic.state_dict()}
             for k, v in self.agents.items()},
            os.path.join(path, "ddpg.pt")
        )

    def load(self, path):
        fp = path if path.endswith(".pt") else os.path.join(path, "ddpg.pt")
        state = torch.load(fp, map_location="cpu", weights_only=False)
        for k in self.agents:
            if k in state:
                self.agents[k].actor.load_state_dict(state[k]["actor"])
                self.agents[k].critic.load_state_dict(state[k]["critic"])
