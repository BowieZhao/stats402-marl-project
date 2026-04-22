from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# Actor-Critic (STABLE)
# =========================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),

            nn.Linear(hidden, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),

            nn.Linear(hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.actor(x), self.critic(x)


# =========================
# SINGLE IPPO AGENT (STABLE CORE)
# =========================
class SingleIPPO:
    def __init__(self, obs_dim, action_dim, cfg):

        self.device = torch.device(cfg.device)

        self.net = ActorCritic(obs_dim, action_dim).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.clip_eps = 0.2

        self.entropy_coef = 0.01
        self.value_coef = 0.5

        self.memory = []

    # =========================
    # OBS NORMALIZATION (CRITICAL FIX)
    # =========================
    def _norm_obs(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)

        return (obs - obs.mean()) / (obs.std() + 1e-8)

    # =========================
    def act(self, obs, explore=True):

        obs = self._norm_obs(obs).unsqueeze(0).to(self.device)

        logits, value = self.net(obs)

        # clamp logits (PREVENT INF SOFTMAX)
        logits = torch.clamp(logits, -10, 10)

        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample() if explore else torch.argmax(logits, dim=-1)

        log_prob = dist.log_prob(action)

        return (
            action.item(),
            log_prob.detach(),
            value.squeeze(-1).detach()
        )

    # =========================
    def store(self, transition):
        self.memory.append(transition)

    # =========================
    def compute_returns(self):
        G = 0.0
        returns = []

        for t in reversed(self.memory):
            r = float(t["reward"])
            done = float(t["done"])

            # reward clip (IMPORTANT)
            r = np.clip(r, -10, 10)

            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)

        return torch.tensor(returns, dtype=torch.float32)

    # =========================
    def update(self):

        if len(self.memory) < 10:
            self.memory.clear()
            return {"actor_loss": 0, "critic_loss": 0, "entropy": 0}

        obs = torch.stack([
            self._norm_obs(t["obs"]) for t in self.memory
        ]).to(self.device)

        actions = torch.tensor(
            [t["action"] for t in self.memory],
            dtype=torch.long
        ).to(self.device)

        old_log_probs = torch.stack([
            t["log_prob"] for t in self.memory
        ]).to(self.device)

        returns = self.compute_returns().to(self.device)

        logits, values = self.net(obs)

        values = torch.clamp(values.squeeze(-1), -10, 10)

        dist = torch.distributions.Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # =========================
        # ADVANTAGE STABILITY (CRITICAL)
        # =========================
        advantages = returns - values.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages = torch.clamp(advantages, -5, 5)

        # =========================
        # PPO RATIO SAFETY
        # =========================
        log_ratio = log_probs - old_log_probs

        log_ratio = torch.clamp(log_ratio, -10, 10)

        ratio = torch.exp(log_ratio)

        # =========================
        # LOSS
        # =========================
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = nn.MSELoss()(values, returns)

        loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy
        )

        # =========================
        # BACKWARD SAFETY
        # =========================
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

        self.optimizer.step()

        self.memory.clear()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
        }


# =========================
# WRAPPER (UNCHANGED LOGIC)
# =========================
class IPPOAgent:
    def __init__(self, config, env):
        self.cfg = config
        self.env = env

        self.agents = {}

        self.adv_agents = []
        self.good_agents = []

        for name in env.possible_agents:
            spec = env.get_agent_spec(name)

            self.agents[name] = SingleIPPO(
                spec.obs_dim,
                spec.action_dim,
                config
            )

            if "adversary" in name:
                self.adv_agents.append(name)
            else:
                self.good_agents.append(name)

    def select_actions(self, obs, env=None, explore=True):
        actions = {}

        for name, ob in obs.items():
            a, logp, val = self.agents[name].act(ob, explore)

            actions[name] = a

            self.agents[name].store({
                "obs": ob,
                "action": a,
                "log_prob": logp,
                "value": val,
                "reward": 0.0,
                "done": 0.0
            })

        return actions

    def observe(self, transition):
        rewards = transition["rewards"]
        dones = transition["terminated"]

        for name in rewards:
            if len(self.agents[name].memory) > 0:
                self.agents[name].memory[-1]["reward"] = float(rewards[name])
                self.agents[name].memory[-1]["done"] = float(dones.get(name, False))

    def update(self, global_step=None):

        adv_a, adv_c, adv_e = [], [], []
        good_a, good_c, good_e = [], [], []

        for name, agent in self.agents.items():
            out = agent.update()

            if "adversary" in name:
                adv_a.append(out["actor_loss"])
                adv_c.append(out["critic_loss"])
                adv_e.append(out["entropy"])
            else:
                good_a.append(out["actor_loss"])
                good_c.append(out["critic_loss"])
                good_e.append(out["entropy"])

        return {
            "adversary/actor_loss": np.mean(adv_a) if adv_a else 0,
            "adversary/critic_loss": np.mean(adv_c) if adv_c else 0,
            "adversary/entropy": np.mean(adv_e) if adv_e else 0,

            "good/actor_loss": np.mean(good_a) if good_a else 0,
            "good/critic_loss": np.mean(good_c) if good_c else 0,
            "good/entropy": np.mean(good_e) if good_e else 0,
        }

    def end_episode(self):
        pass