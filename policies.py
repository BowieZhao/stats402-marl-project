from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import to_tensor


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Transition:
    obs: np.ndarray
    critic_input: np.ndarray
    action: int
    log_prob: float
    reward: float
    done: float
    value: float
    next_critic_input: np.ndarray


class PPOBuffer:
    def __init__(self):
        self.storage: list[Transition] = []

    def add(self, transition: Transition):
        self.storage.append(transition)

    def clear(self):
        self.storage.clear()

    def __len__(self):
        return len(self.storage)


class BasePPOPolicy:
    def __init__(self, obs_dim: int, action_dim: int, critic_input_dim: int, cfg):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic_input_dim = critic_input_dim
        self.cfg = cfg
        self.device = cfg.device

        self.actor = ActorNetwork(obs_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.critic = CriticNetwork(critic_input_dim, cfg.hidden_dim).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def get_critic_input(self, obs_dict: dict, agent_id: str, agent_order: list[str]) -> np.ndarray:
        raise NotImplementedError

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, critic_input: np.ndarray):
        """
        Training-time action selection: stochastic sampling.
        """
        obs_t = to_tensor(obs, self.device).unsqueeze(0)
        critic_t = to_tensor(critic_input, self.device).unsqueeze(0)

        logits = self.actor(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(critic_t)

        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray, critic_input: np.ndarray):
        """
        Eval / visualization action selection: deterministic argmax.
        """
        obs_t = to_tensor(obs, self.device).unsqueeze(0)
        critic_t = to_tensor(critic_input, self.device).unsqueeze(0)

        logits = self.actor(obs_t)
        action = torch.argmax(logits, dim=-1)
        value = self.critic(critic_t)

        return int(action.item()), float(value.item())

    def evaluate_actions(self, obs_batch: torch.Tensor, critic_batch: torch.Tensor, action_batch: torch.Tensor):
        logits = self.actor(obs_batch)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action_batch)
        entropy = dist.entropy().mean()
        values = self.critic(critic_batch)
        return log_probs, entropy, values

    def compute_gae(self, rewards, dones, values, next_values):
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]

        adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return adv, ret

    def update(self, buffer: PPOBuffer):
        if len(buffer) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        obs = np.stack([t.obs for t in buffer.storage], axis=0)
        critic_inputs = np.stack([t.critic_input for t in buffer.storage], axis=0)
        actions = np.array([t.action for t in buffer.storage], dtype=np.int64)
        old_log_probs = np.array([t.log_prob for t in buffer.storage], dtype=np.float32)
        rewards = np.array([t.reward for t in buffer.storage], dtype=np.float32)
        dones = np.array([t.done for t in buffer.storage], dtype=np.float32)
        values = np.array([t.value for t in buffer.storage], dtype=np.float32)
        next_critic_inputs = np.stack([t.next_critic_input for t in buffer.storage], axis=0)

        with torch.no_grad():
            next_values = self.critic(to_tensor(next_critic_inputs, self.device)).cpu().numpy()

        advantages, returns = self.compute_gae(rewards, dones, values, next_values)

        obs_t = to_tensor(obs, self.device)
        critic_t = to_tensor(critic_inputs, self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)

        n = obs_t.size(0)
        batch_size = min(self.cfg.minibatch_size, n)

        actor_losses = []
        critic_losses = []
        entropies = []

        for _ in range(self.cfg.ppo_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                mb_idx = indices[start:start + batch_size]

                mb_obs = obs_t[mb_idx]
                mb_critic = critic_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_log_probs, entropy, new_values = self.evaluate_actions(mb_obs, mb_critic, mb_actions)
                ratio = (new_log_probs - mb_old_log_probs).exp()

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.cfg.clip_eps,
                    1.0 + self.cfg.clip_eps,
                ) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values, mb_returns)
                loss = actor_loss + self.cfg.value_coef * critic_loss - self.cfg.entropy_coef * entropy

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.item()))

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])


class IPPOPolicy(BasePPOPolicy):
    def get_critic_input(self, obs_dict: dict, agent_id: str, agent_order: list[str]) -> np.ndarray:
        return np.asarray(obs_dict[agent_id], dtype=np.float32)


class MAPPOPolicy(BasePPOPolicy):
    def get_critic_input(self, obs_dict: dict, agent_id: str, agent_order: list[str]) -> np.ndarray:
        parts = [np.asarray(obs_dict[a], dtype=np.float32).reshape(-1) for a in agent_order]
        return np.concatenate(parts, axis=0)


def build_policy(algo: str, obs_dim: int, action_dim: int, global_state_dim: int, cfg):
    algo = algo.lower()
    if algo == "ippo":
        return IPPOPolicy(obs_dim, action_dim, obs_dim, cfg)
    if algo == "mappo":
        return MAPPOPolicy(obs_dim, action_dim, global_state_dim, cfg)
    raise ValueError(f"Unsupported algo: {algo}")