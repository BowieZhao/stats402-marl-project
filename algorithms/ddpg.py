"""
Continuous-action DDPG baseline for simple_world_comm with continuous_actions=True.

Action spaces:
  - leader: Box(0, 1, (9,))   -> 5 movement + 4 communication channels
  - normal adv / good: Box(0, 1, (5,))

Design (matches MAPPO's RoleGroup structure):
  - 3 actor-critic pairs: leader, normal_adversary (shared across 3), good
  - Role-shared replay buffers
  - OU noise for exploration (decays over training)
  - Target policy smoothing (TD3-style noise on target action) for stability
  - Output: sigmoid -> [0, 1] matching mpe2 continuous Box specs
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
# Ornstein-Uhlenbeck noise
# ═══════════════════════════════════════════════════════════
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state.copy()


# ═══════════════════════════════════════════════════════════
# Networks
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
        # mpe2 Box action space is [0, 1] per component -> sigmoid
        return torch.sigmoid(self.net(x))


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
# Replay buffer
# ═══════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buf = deque(maxlen=capacity)

    def add(self, obs, act, rew, next_obs, done):
        self.buf.append((obs, act, rew, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(np.array(act), dtype=torch.float32),
            torch.tensor(rew, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buf)


# ═══════════════════════════════════════════════════════════
# Per-role learner
# ═══════════════════════════════════════════════════════════
class RoleDDPGLearner:
    def __init__(self, obs_dim, action_dim, cfg, name=""):
        self.device = torch.device(cfg.device)
        self.name = name
        self.action_dim = action_dim

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_tgt = Actor(obs_dim, action_dim).to(self.device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, action_dim).to(self.device)
        self.critic_tgt = Critic(obs_dim, action_dim).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 128
        self.warmup_steps = 2000
        # TD3-style: add small noise to target action for smoothing
        self.target_noise = 0.1
        self.target_noise_clip = 0.3

        # OU exploration noise; one instance per agent_name (since multiple
        # agents share this learner, each maintains its own OU state)
        self.noises: Dict[str, OUNoise] = {}
        self.noise_scale = 1.0
        self.noise_min = 0.05

        self.buffer = ReplayBuffer(capacity=200000)

    def _get_noise(self, agent_name):
        if agent_name not in self.noises:
            self.noises[agent_name] = OUNoise(self.action_dim,
                                               sigma=0.2)
        return self.noises[agent_name]

    def act(self, obs, agent_name, explore=True):
        obs_t = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(obs_t).cpu().numpy()[0]
        if explore:
            n = self._get_noise(agent_name).sample()
            a = a + self.noise_scale * n
            a = np.clip(a, 0.0, 1.0)
        return a.astype(np.float32)

    def reset_noise(self, agent_name):
        if agent_name in self.noises:
            self.noises[agent_name].reset()

    def update(self):
        if len(self.buffer) < max(self.warmup_steps, self.batch_size):
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size)
        obs = obs.to(self.device)
        act = act.to(self.device)
        rew = rew.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        # Critic update with target policy smoothing
        with torch.no_grad():
            next_act = self.actor_tgt(next_obs)
            noise = (torch.randn_like(next_act) * self.target_noise).clamp(
                -self.target_noise_clip, self.target_noise_clip)
            next_act = (next_act + noise).clamp(0.0, 1.0)
            target_q = self.critic_tgt(next_obs, next_act)
            y = rew + self.gamma * (1.0 - done) * target_q

        q = self.critic(obs, act)
        critic_loss = F.smooth_l1_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Actor update
        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Polyak soft updates
        for p, tp in zip(self.actor.parameters(), self.actor_tgt.parameters()):
            tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
        for p, tp in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

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

        # Sanity check: this DDPG requires continuous actions
        if not getattr(cfg, "continuous_actions", False):
            raise RuntimeError(
                "DDPGAgent requires config.continuous_actions=True. "
                "main.py auto-sets this for algo='ddpg'."
            )

        self.leader_names = [a for a in env.possible_agents if "leadadversary" in a]
        self.normal_adv_names = [a for a in env.possible_agents
                                  if "adversary" in a and "leadadversary" not in a]
        self.good_names = [a for a in env.possible_agents if a.startswith("agent_")]

        self.adv_agents = self.leader_names + self.normal_adv_names
        self.good_agents = self.good_names

        self.learners: Dict[str, RoleDDPGLearner] = {}
        if self.leader_names:
            spec = env.get_agent_spec(self.leader_names[0])
            self.learners["leader"] = RoleDDPGLearner(spec.obs_dim, spec.action_dim, cfg, "leader")
        if self.normal_adv_names:
            spec = env.get_agent_spec(self.normal_adv_names[0])
            self.learners["adversary"] = RoleDDPGLearner(spec.obs_dim, spec.action_dim, cfg, "adversary")
        if self.good_names:
            spec = env.get_agent_spec(self.good_names[0])
            self.learners["good"] = RoleDDPGLearner(spec.obs_dim, spec.action_dim, cfg, "good")

        # Noise scale decay (per episode)
        self.noise_decay = 0.999

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
            actions[name] = self.learners[role].act(ob, name, explore=explore)
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
            self.learners[role].buffer.add(
                np.asarray(obs[a], dtype=np.float32),
                np.asarray(actions[a], dtype=np.float32),
                float(rewards[a]),
                np.asarray(next_obs[a], dtype=np.float32),
                done
            )

    def end_episode(self):
        # Reset OU noise per agent at episode end + decay scale
        for learner in self.learners.values():
            for name in list(learner.noises.keys()):
                learner.reset_noise(name)
            learner.noise_scale = max(learner.noise_scale * self.noise_decay,
                                       learner.noise_min)

    def update(self, global_step=None):
        logs = {}
        n_updates = 4
        for role, learner in self.learners.items():
            a_losses, c_losses = [], []
            for _ in range(n_updates):
                out = learner.update()
                if out["actor_loss"] != 0:
                    a_losses.append(out["actor_loss"])
                if out["critic_loss"] != 0:
                    c_losses.append(out["critic_loss"])
            if a_losses:
                logs[f"{role}/actor_loss"] = float(np.mean(a_losses))
            if c_losses:
                logs[f"{role}/critic_loss"] = float(np.mean(c_losses))
        if self.learners:
            logs["noise_scale"] = next(iter(self.learners.values())).noise_scale
        return logs

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        state = {role: {"actor": learner.actor.state_dict(),
                        "critic": learner.critic.state_dict()}
                 for role, learner in self.learners.items()}
        torch.save(state, os.path.join(path, "ddpg.pt"))

    def load(self, path):
        fp = path if path.endswith(".pt") else os.path.join(path, "ddpg.pt")
        state = torch.load(fp, map_location="cpu", weights_only=False)
        for role, sd in state.items():
            if role in self.learners:
                self.learners[role].actor.load_state_dict(sd["actor"])
                self.learners[role].critic.load_state_dict(sd["critic"])
