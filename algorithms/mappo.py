"""
MAPPO with FINAL design:
  - Leader: standard ActorNet (action over 20-dim joint move x message)
  - Normal Adversary: AdversaryAlphaActor (own_logits + alpha + plan_bias mixing)
  - Good: standard ActorNet (or frozen)
  - All roles use parameter sharing within role group
"""

from __future__ import annotations
import os
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# ═══════════════════════════════════════════════════════════════
# Running mean/std for advantage normalization
# ═══════════════════════════════════════════════════════════════
class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        if x.size == 0:
            return
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = x.size
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var = M2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(math.sqrt(max(self.var, 1e-8)))


def _mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


# ═══════════════════════════════════════════════════════════════
# Standard ActorNet (used for leader and good)
# ═══════════════════════════════════════════════════════════════
class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = _mlp(obs_dim, action_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        logits = self.net(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy()

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits = self.net(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


# ═══════════════════════════════════════════════════════════════
# Adversary actor with alpha head + plan_bias mixing
# ═══════════════════════════════════════════════════════════════
class AdversaryAlphaActor(nn.Module):
    """
    Outputs (own_logits, alpha) given augmented adv obs.
    The actual final logits are computed as:
        final = alpha * own_logits + (1 - alpha) * plan_bias
    plan_bias is computed externally (from env) and passed in for action sampling
    AND for evaluate_actions().

    Trunk produces a hidden representation; two heads then produce own_logits
    and alpha (sigmoid-bounded scalar).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128,
                 alpha_init: float = 0.5, alpha_learnable: bool = True):
        super().__init__()
        self.action_dim = action_dim
        self.alpha_learnable = alpha_learnable
        self.alpha_init = alpha_init

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.logits_head = nn.Linear(hidden, action_dim)

        if alpha_learnable:
            self.alpha_head = nn.Linear(hidden, 1)
            # Initialize bias so initial alpha ≈ alpha_init
            #   alpha = sigmoid(b) = alpha_init → b = log(alpha_init / (1 - alpha_init))
            init_bias = math.log(alpha_init / (1.0 - alpha_init))
            with torch.no_grad():
                self.alpha_head.bias.fill_(init_bias)
                self.alpha_head.weight.zero_()
        else:
            self.alpha_head = None  # alpha will be fixed at alpha_init

    def _compute_components(self, obs: torch.Tensor):
        """Returns (own_logits, alpha)."""
        h = self.trunk(obs)
        own_logits = self.logits_head(h)
        if self.alpha_learnable:
            alpha = torch.sigmoid(self.alpha_head(h))
        else:
            alpha = torch.full(
                (obs.shape[0], 1), self.alpha_init,
                dtype=obs.dtype, device=obs.device,
            )
        return own_logits, alpha

    def forward(self, obs: torch.Tensor,
                plan_bias: torch.Tensor) -> torch.Tensor:
        """Returns final_logits = alpha * own_logits + (1-alpha) * plan_bias"""
        own_logits, alpha = self._compute_components(obs)
        final_logits = alpha * own_logits + (1.0 - alpha) * plan_bias
        return final_logits

    def get_action(self, obs: torch.Tensor, plan_bias: torch.Tensor,
                   deterministic: bool = False):
        """Sample an action using final_logits."""
        own_logits, alpha = self._compute_components(obs)
        final_logits = alpha * own_logits + (1.0 - alpha) * plan_bias
        dist = Categorical(logits=final_logits)
        if deterministic:
            action = final_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy(), alpha

    def evaluate_actions(self, obs: torch.Tensor, plan_bias: torch.Tensor,
                          actions: torch.Tensor):
        own_logits, alpha = self._compute_components(obs)
        final_logits = alpha * own_logits + (1.0 - alpha) * plan_bias
        dist = Categorical(logits=final_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


# ═══════════════════════════════════════════════════════════════
# Critic (shared per role)
# ═══════════════════════════════════════════════════════════════
class CriticNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = _mlp(obs_dim, 1, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# Role group: shared actor + critic for one role
# ═══════════════════════════════════════════════════════════════
class RoleGroup:
    def __init__(self, name: str, agent_ids: List[str],
                 obs_dim: int, action_dim: int, config,
                 device: torch.device,
                 is_adversary_role: bool = False):
        self.name = name
        self.agent_ids = agent_ids
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.is_adversary_role = is_adversary_role

        if is_adversary_role:
            self.actor = AdversaryAlphaActor(
                obs_dim, action_dim,
                alpha_init=config.alpha_init,
                alpha_learnable=config.alpha_is_learnable,
            ).to(device)
        else:
            self.actor = ActorNet(obs_dim, action_dim).to(device)

        self.critic = CriticNet(obs_dim).to(device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        self.adv_rms = RunningMeanStd()


# ═══════════════════════════════════════════════════════════════
# MAPPOAgent: ties everything together
# ═══════════════════════════════════════════════════════════════
class MAPPOAgent:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.device = torch.device(config.device)

        # Build role groups
        self.role_groups: Dict[str, RoleGroup] = {}
        self._build_role_groups(env)

        # Episode buffers per agent
        self._buffers: Dict[str, list] = {a: [] for a in env.possible_agents}

    def _build_role_groups(self, env):
        leader_ids = [a for a in env.possible_agents if "leadadversary" in a]
        adv_ids = [a for a in env.possible_agents
                   if "adversary" in a and "leadadversary" not in a]
        good_ids = [a for a in env.possible_agents if a.startswith("agent_")]

        if leader_ids:
            spec = env.get_agent_spec(leader_ids[0])
            self.role_groups["leader"] = RoleGroup(
                "leader", leader_ids, spec.obs_dim, spec.action_dim,
                self.config, self.device,
                is_adversary_role=False,
            )
        if adv_ids:
            spec = env.get_agent_spec(adv_ids[0])
            self.role_groups["adversary"] = RoleGroup(
                "adversary", adv_ids, spec.obs_dim, spec.action_dim,
                self.config, self.device,
                is_adversary_role=True,
            )
        if good_ids:
            spec = env.get_agent_spec(good_ids[0])
            self.role_groups["good"] = RoleGroup(
                "good", good_ids, spec.obs_dim, spec.action_dim,
                self.config, self.device,
                is_adversary_role=False,
            )

    def _role_of(self, agent_name: str) -> Optional[str]:
        if "leadadversary" in agent_name:
            return "leader"
        if "adversary" in agent_name:
            return "adversary"
        if agent_name.startswith("agent_"):
            return "good"
        return None

    # ─── Action selection ───────────────────────────────────────
    def select_actions(self, obs_dict: dict, env=None,
                       explore: bool = True) -> dict:
        """Returns {agent_name: action_int}."""
        actions = {}
        with torch.no_grad():
            for agent_name, obs in obs_dict.items():
                role = self._role_of(agent_name)
                if role is None or role not in self.role_groups:
                    continue
                grp = self.role_groups[role]

                obs_t = torch.tensor(obs, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)

                if grp.is_adversary_role:
                    # Need plan_bias; current message is in obs[34:38] one-hot
                    msg_onehot = obs[34:38]
                    if msg_onehot.sum() < 0.5:
                        # No message available (E2 setting or first step)
                        message = 0
                    else:
                        message = int(np.argmax(msg_onehot))
                    plan_bias = self.env.compute_plan_bias(message, obs)
                    plan_bias_t = torch.tensor(
                        plan_bias, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                    action, _, _, _ = grp.actor.get_action(
                        obs_t, plan_bias_t, deterministic=not explore,
                    )
                else:
                    action, _, _ = grp.actor.get_action(
                        obs_t, deterministic=not explore,
                    )
                actions[agent_name] = int(action.item())
        return actions

    # ─── Experience collection ──────────────────────────────────
    def observe(self, transition: dict):
        """
        transition keys: obs, actions, rewards, next_obs, terminated, truncated
        """
        obs = transition["obs"]
        actions = transition["actions"]
        rewards = transition["rewards"]
        next_obs = transition["next_obs"]
        terms = transition.get("terminated", {})
        truncs = transition.get("truncated", {})

        # Process rewards through (optional) per-role normalization
        rewards = self._process_rewards(rewards)

        for agent_name, ob in obs.items():
            role = self._role_of(agent_name)
            if role is None or role not in self.role_groups:
                continue
            grp = self.role_groups[role]

            # Compute log_prob and value at storage time
            obs_t = torch.tensor(ob, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            action_t = torch.tensor([actions[agent_name]],
                                     dtype=torch.long, device=self.device)

            with torch.no_grad():
                if grp.is_adversary_role:
                    msg_onehot = ob[34:38]
                    if msg_onehot.sum() < 0.5:
                        message = 0
                    else:
                        message = int(np.argmax(msg_onehot))
                    plan_bias = self.env.compute_plan_bias(message, ob)
                    plan_bias_t = torch.tensor(
                        plan_bias, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                    log_prob, _ = grp.actor.evaluate_actions(
                        obs_t, plan_bias_t, action_t
                    )
                    value = grp.critic(obs_t)
                    plan_bias_for_storage = plan_bias  # numpy
                else:
                    log_prob, _ = grp.actor.evaluate_actions(obs_t, action_t)
                    value = grp.critic(obs_t)
                    plan_bias_for_storage = None

            done = float(terms.get(agent_name, False) or truncs.get(agent_name, False))

            self._buffers[agent_name].append({
                "obs": np.asarray(ob, dtype=np.float32),
                "action": int(actions[agent_name]),
                "reward": float(rewards[agent_name]),
                "next_obs": np.asarray(next_obs[agent_name], dtype=np.float32),
                "log_prob": float(log_prob.item()),
                "value": float(value.item()),
                "done": done,
                "plan_bias": plan_bias_for_storage,  # only for advs
            })

    def end_episode(self):
        """No-op; handled when buffers are flushed in update()."""
        pass

    def _process_rewards(self, raw: dict) -> dict:
        """Hook for reward normalization or shaping. Currently identity."""
        return raw

    # ─── PPO update ─────────────────────────────────────────────
    def update(self, global_step: int) -> dict:
        logs = {}
        for role_name, grp in self.role_groups.items():
            if role_name == "good" and self.config.frozen_good_path is not None:
                # Skip update for frozen good
                continue
            log = self._update_group(grp)
            for k, v in log.items():
                logs[f"{role_name}/{k}"] = v
        # Clear buffers after update
        for a in self._buffers:
            self._buffers[a] = []
        return logs

    def _update_group(self, grp: RoleGroup) -> dict:
        # Aggregate buffer data across all agents in this group
        all_obs, all_acts, all_lps, all_advs, all_rets = [], [], [], [], []
        all_plan_bias = []  # only used for adversary role

        for agent_id in grp.agent_ids:
            buf = self._buffers.get(agent_id, [])
            if not buf:
                continue
            obs_arr = np.stack([t["obs"] for t in buf])
            acts_arr = np.array([t["action"] for t in buf], dtype=np.int64)
            lps_arr = np.array([t["log_prob"] for t in buf], dtype=np.float32)
            rews_arr = np.array([t["reward"] for t in buf], dtype=np.float32)
            vals_arr = np.array([t["value"] for t in buf], dtype=np.float32)
            dones_arr = np.array([t["done"] for t in buf], dtype=np.float32)

            # Bootstrap value for last state
            with torch.no_grad():
                last_obs = torch.tensor(buf[-1]["next_obs"],
                                          dtype=torch.float32,
                                          device=self.device).unsqueeze(0)
                bootstrap = float(grp.critic(last_obs).item())
            if dones_arr[-1] > 0:
                bootstrap = 0.0

            adv, ret = self._compute_gae(
                rews_arr, vals_arr, dones_arr, bootstrap,
                self.config.gamma, self.config.gae_lambda
            )

            all_obs.append(obs_arr)
            all_acts.append(acts_arr)
            all_lps.append(lps_arr)
            all_advs.append(adv)
            all_rets.append(ret)

            if grp.is_adversary_role:
                pb = np.stack([t["plan_bias"] for t in buf])
                all_plan_bias.append(pb)

        if not all_obs:
            return {}

        obs_cat = np.concatenate(all_obs, axis=0)
        acts_cat = np.concatenate(all_acts, axis=0)
        lps_cat = np.concatenate(all_lps, axis=0)
        adv_cat = np.concatenate(all_advs, axis=0)
        ret_cat = np.concatenate(all_rets, axis=0)
        plan_bias_cat = np.concatenate(all_plan_bias, axis=0) if all_plan_bias else None

        # Advantage normalization
        grp.adv_rms.update(adv_cat)
        adv_cat = (adv_cat - grp.adv_rms.mean) / (grp.adv_rms.std + 1e-8)

        log = self._ppo_epochs(grp, obs_cat, acts_cat, lps_cat,
                                adv_cat, ret_cat, plan_bias_cat)
        return log

    @staticmethod
    def _compute_gae(rewards, values, dones, bootstrap, gamma, gae_lam):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        next_val = bootstrap
        for t in reversed(range(T)):
            non_term = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * non_term - values[t]
            last_gae = delta + gamma * gae_lam * non_term * last_gae
            adv[t] = last_gae
            next_val = values[t]
        ret = adv + values
        return adv, ret

    def _ppo_epochs(self, grp, obs, acts, lps_old, adv, ret,
                     plan_bias=None) -> dict:
        device = self.device
        N = obs.shape[0]
        batch_size = self.config.minibatch_size
        epochs = self.config.ppo_epochs

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        acts_t = torch.tensor(acts, dtype=torch.long, device=device)
        lps_old_t = torch.tensor(lps_old, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
        if plan_bias is not None:
            plan_bias_t = torch.tensor(plan_bias, dtype=torch.float32, device=device)
        else:
            plan_bias_t = None

        actor_losses, critic_losses, entropies = [], [], []

        indices = np.arange(N)
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = indices[start:end]
                idx_t = torch.tensor(idx, dtype=torch.long, device=device)

                b_obs = obs_t[idx_t]
                b_acts = acts_t[idx_t]
                b_lps_old = lps_old_t[idx_t]
                b_adv = adv_t[idx_t]
                b_ret = ret_t[idx_t]

                if grp.is_adversary_role and plan_bias_t is not None:
                    b_pb = plan_bias_t[idx_t]
                    new_log_prob, entropy = grp.actor.evaluate_actions(
                        b_obs, b_pb, b_acts
                    )
                else:
                    new_log_prob, entropy = grp.actor.evaluate_actions(
                        b_obs, b_acts
                    )

                ratio = torch.exp(new_log_prob - b_lps_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip,
                                     1 + self.config.ppo_clip) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = actor_loss - self.config.entropy_coef * entropy.mean()

                value = grp.critic(b_obs)
                critic_loss = F.mse_loss(value, b_ret)

                grp.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(grp.actor.parameters(), 0.5)
                grp.actor_opt.step()

                grp.critic_opt.zero_grad()
                (self.config.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(grp.critic.parameters(), 0.5)
                grp.critic_opt.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.mean().item()))

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    # ─── Save/load ──────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for role_name, grp in self.role_groups.items():
            torch.save(grp.actor.state_dict(),
                       os.path.join(path, f"{role_name}.pt"))
            torch.save(grp.critic.state_dict(),
                       os.path.join(path, f"{role_name}_critic.pt"))

    def load(self, path: str):
        for role_name, grp in self.role_groups.items():
            actor_path = os.path.join(path, f"{role_name}.pt")
            critic_path = os.path.join(path, f"{role_name}_critic.pt")
            if os.path.exists(actor_path):
                grp.actor.load_state_dict(torch.load(actor_path,
                                                      map_location=self.device,
                                                      weights_only=False),
                                           strict=False)
            if os.path.exists(critic_path):
                grp.critic.load_state_dict(torch.load(critic_path,
                                                       map_location=self.device,
                                                       weights_only=False),
                                            strict=False)
