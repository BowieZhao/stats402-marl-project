"""
MAPPO-variant for simple_world_comm_v3 (discrete actions).

Design principles
─────────────────
- Three RoleGroups: leader (solo), normal adversary (shared params), good (shared params)
- ALL actors and critics use ONLY local observation — no centralized critic
- Adversary side receives team-mean reward → cooperative objective
- Good side receives individual reward
- Trajectories stored per-episode; GAE computed within each episode (no cross-episode leakage)
- Value normalization: normalise critic targets; denormalise stored values for GAE
- Reward normalization: divide by running std of per-side rewards

Call protocol (from ExperimentRunner)
─────────────────────────────────────
    actions = agent.select_actions(obs_dict, env, explore=True)
    # env.step(...)
    agent.observe(transition)
    # ... repeat for all steps in episode ...
    agent.end_episode()
    # ... repeat for N episodes ...
    metrics = agent.update(global_step)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict

from envs import is_leader, is_normal_adversary, is_good


# ═══════════════════════════════════════════════════════════════
# Running statistics (Welford online algorithm)
# ═══════════════════════════════════════════════════════════════

class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4):
        self.mean = np.float64(0.0)
        self.var = np.float64(1.0)
        self.count = np.float64(epsilon)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).flatten()
        if x.size == 0:
            return
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.size
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + 1e-8))


# ═══════════════════════════════════════════════════════════════
# Networks
# ═══════════════════════════════════════════════════════════════

def _mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = _mlp(obs_dim, action_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


class CriticNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = _mlp(obs_dim, 1, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# RoleGroup: one (actor, critic) pair shared by same-role agents
# ═══════════════════════════════════════════════════════════════

class RoleGroup:
    def __init__(self, name: str, agent_ids: list,
                 obs_dim: int, action_dim: int,
                 config, device: torch.device):
        self.name = name
        self.agent_ids = list(agent_ids)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        h = config.hidden_dim
        self.actor = ActorNet(obs_dim, action_dim, h).to(device)
        self.critic = CriticNet(obs_dim, h).to(device)

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr, eps=1e-5)

        # Per-role running stats for value (return) normalisation
        self.value_rms = RunningMeanStd()


# ═══════════════════════════════════════════════════════════════
# MAPPOAgent
# ═══════════════════════════════════════════════════════════════

class MAPPOAgent:

    def __init__(self, config, env):
        self.config = config
        self.device = torch.device(config.device)

        # ── identify agents and roles ────────────────────────────
        self.all_agents = env.possible_agents
        self.adv_agents = [a for a in self.all_agents if is_leader(a) or is_normal_adversary(a)]
        self.good_agents = [a for a in self.all_agents if is_good(a)]

        # ── build role groups ────────────────────────────────────
        self.role_groups: dict[str, RoleGroup] = {}
        self._build_role_groups(env)

        # Map each agent → its RoleGroup
        self.agent_to_group: dict[str, RoleGroup] = {}
        for grp in self.role_groups.values():
            for aid in grp.agent_ids:
                self.agent_to_group[aid] = grp

        # ── per-side reward running stats ────────────────────────
        self.adv_reward_rms = RunningMeanStd()
        self.good_reward_rms = RunningMeanStd()

        # ── trajectory storage (episode-aware) ───────────────────
        # Each episode is a list of step dicts.
        self._episodes: list[list[dict]] = []
        self._current_ep: list[dict] = []

        # Temporaries set by select_actions, consumed by observe
        self._last_log_probs: dict[str, float] = {}
        self._last_values: dict[str, float] = {}

    # ──────────────────────────────────────────────────────────
    # Role group construction
    # ──────────────────────────────────────────────────────────

    def _build_role_groups(self, env):
        cfg = self.config
        obs_spaces = env.observation_spaces
        act_spaces = env.action_spaces

        leaders = [a for a in self.all_agents if is_leader(a)]
        normals = [a for a in self.all_agents if is_normal_adversary(a)]
        goods   = [a for a in self.all_agents if is_good(a)]

        if leaders:
            ref = leaders[0]
            self.role_groups["leader"] = RoleGroup(
                "leader", leaders,
                obs_spaces[ref].shape[0],
                act_spaces[ref].n,     # 20 for discrete leader
                cfg, self.device,
            )
        if normals:
            ref = normals[0]
            self.role_groups["adversary"] = RoleGroup(
                "adversary", normals,
                obs_spaces[ref].shape[0],
                act_spaces[ref].n,     # 5 for normal adversary
                cfg, self.device,
            )
        if goods:
            ref = goods[0]
            self.role_groups["good"] = RoleGroup(
                "good", goods,
                obs_spaces[ref].shape[0],
                act_spaces[ref].n,     # 5 for good agent
                cfg, self.device,
            )

    # ──────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────

    def select_actions(self, obs_dict: dict, env, explore: bool = True) -> dict:
        """
        Returns {agent_id: action_int} for every agent in obs_dict.
        Caches log_probs and values for observe().
        """
        actions = {}
        log_probs = {}
        values = {}

        for agent_id, obs in obs_dict.items():
            grp = self.agent_to_group.get(agent_id)
            if grp is None:
                # Agent not managed by us — shouldn't happen, but be safe
                continue
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, lp, _ = grp.actor.get_action(obs_t, deterministic=not explore)
                val = grp.critic(obs_t)
            actions[agent_id] = action.item()
            log_probs[agent_id] = lp.item()
            values[agent_id] = val.item()

        self._last_log_probs = log_probs
        self._last_values = values
        return actions

    # ──────────────────────────────────────────────────────────
    # Transition storage
    # ──────────────────────────────────────────────────────────

    def observe(self, transition: dict):
        """
        Store one step.  Expected keys in transition:
            obs, actions, rewards, next_obs, terminated, truncated
        All dicts keyed by agent_id.
        """
        obs = transition["obs"]
        rewards_raw = transition["rewards"]
        next_obs = transition["next_obs"]
        terminated = transition["terminated"]
        truncated = transition["truncated"]

        processed_rewards = self._process_rewards(rewards_raw)

        step = {
            "obs":        {k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                           for k, v in obs.items() if k in self.agent_to_group},
            "actions":    {k: v for k, v in transition["actions"].items()
                           if k in self.agent_to_group},
            "log_probs":  {k: v for k, v in self._last_log_probs.items()},
            "values":     {k: v for k, v in self._last_values.items()},
            "rewards":    processed_rewards,
            "terminated": {k: float(v) for k, v in terminated.items()
                           if k in self.agent_to_group},
            "truncated":  {k: float(v) for k, v in truncated.items()
                           if k in self.agent_to_group},
            "next_obs":   {k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                           for k, v in next_obs.items() if k in self.agent_to_group},
        }
        self._current_ep.append(step)

    def end_episode(self):
        """Called by the runner after each episode finishes."""
        if self._current_ep:
            self._episodes.append(self._current_ep)
            self._current_ep = []

    # ──────────────────────────────────────────────────────────
    # Reward processing (no information leakage)
    # ──────────────────────────────────────────────────────────

    def _process_rewards(self, raw: dict) -> dict:
        """
        Adversary side → team mean, optionally normalised.
        Good side      → individual, optionally normalised.
        """
        cfg = self.config
        processed = {}

        # ── adversary team mean ──────────────────────────────
        adv_raw = [raw[a] for a in self.adv_agents if a in raw]
        if adv_raw:
            team_mean = float(np.mean(adv_raw))
            if cfg.use_reward_normalization:
                self.adv_reward_rms.update(np.array([team_mean]))
                r = team_mean / self.adv_reward_rms.std
            else:
                r = team_mean
            if cfg.use_reward_tanh:
                r = float(np.tanh(r / cfg.reward_tanh_scale))
            for a in self.adv_agents:
                if a in raw:
                    processed[a] = r

        # ── good individual ──────────────────────────────────
        good_vals = [raw[a] for a in self.good_agents if a in raw]
        if good_vals and cfg.use_reward_normalization:
            self.good_reward_rms.update(np.array(good_vals))
        for a in self.good_agents:
            if a in raw:
                r = raw[a]
                if cfg.use_reward_normalization:
                    r = r / self.good_reward_rms.std
                if cfg.use_reward_tanh:
                    r = float(np.tanh(r / cfg.reward_tanh_scale))
                processed[a] = r

        return processed

    # ──────────────────────────────────────────────────────────
    # PPO update
    # ──────────────────────────────────────────────────────────

    def update(self, global_step: int) -> dict:
        if not self._episodes:
            return {}

        all_metrics = {}
        for grp_name, grp in self.role_groups.items():
            m = self._update_group(grp)
            for k, v in m.items():
                all_metrics[f"{grp_name}/{k}"] = v

        self._episodes = []
        return all_metrics

    def _update_group(self, grp: RoleGroup) -> dict:
        """
        For every agent in this group and every stored episode:
          1. Extract per-agent trajectory
          2. Compute GAE within the episode
          3. Concatenate across agents and episodes
          4. Run PPO epochs on the batch
        """
        cfg = self.config
        all_obs, all_acts, all_lps, all_adv, all_ret = [], [], [], [], []

        for episode in self._episodes:
            for agent_id in grp.agent_ids:
                result = self._extract_agent_episode(agent_id, episode, grp)
                if result is None:
                    continue
                obs_t, acts_t, lps_t, adv_t, ret_t = result
                all_obs.append(obs_t)
                all_acts.append(acts_t)
                all_lps.append(lps_t)
                all_adv.append(adv_t)
                all_ret.append(ret_t)

        if not all_obs:
            return {}

        obs_b  = torch.cat(all_obs, 0)
        acts_b = torch.cat(all_acts, 0)
        lps_b  = torch.cat(all_lps, 0)
        adv_b  = torch.cat(all_adv, 0)
        ret_b  = torch.cat(all_ret, 0)

        # Normalise advantages across the whole group batch
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        return self._ppo_epochs(grp, obs_b, acts_b, lps_b, adv_b, ret_b)

    def _extract_agent_episode(self, agent_id: str, episode: list, grp: RoleGroup):
        """
        Extract one agent's data from one episode, compute GAE, return tensors.
        Returns None if the agent has fewer than 2 steps in this episode.
        """
        obs_list, act_list, lp_list, val_list = [], [], [], []
        rew_list, term_list, trunc_list, nobs_list = [], [], [], []

        for step in episode:
            if agent_id not in step["obs"]:
                continue
            obs_list.append(step["obs"][agent_id])
            act_list.append(step["actions"].get(agent_id, 0))
            lp_list.append(step["log_probs"].get(agent_id, 0.0))
            val_list.append(step["values"].get(agent_id, 0.0))
            rew_list.append(step["rewards"].get(agent_id, 0.0))
            term_list.append(step["terminated"].get(agent_id, 0.0))
            trunc_list.append(step["truncated"].get(agent_id, 0.0))
            nobs_list.append(step["next_obs"].get(agent_id, step["obs"][agent_id]))

        T = len(obs_list)
        if T < 1:
            return None

        rews  = np.array(rew_list, dtype=np.float32)
        vals  = np.array(val_list, dtype=np.float32)
        terms = np.array(term_list, dtype=np.float32)

        # ── bootstrap value at episode end ───────────────────
        # If the last step was a true termination → bootstrap = 0.
        # If truncated (max_cycles reached) → bootstrap with V(next_obs).
        last_term = term_list[-1]
        last_trunc = trunc_list[-1]
        if last_term > 0.5:
            bootstrap = 0.0
        elif last_trunc > 0.5:
            with torch.no_grad():
                bootstrap = grp.critic(nobs_list[-1].unsqueeze(0)).item()
        else:
            # Mid-episode (shouldn't happen with end_episode, but be safe)
            bootstrap = val_list[-1]

        # ── value denormalisation for GAE ────────────────────
        if self.config.use_value_normalization and grp.value_rms.count > 1.0:
            vals_denorm = vals * grp.value_rms.std + grp.value_rms.mean
            bootstrap_denorm = bootstrap * grp.value_rms.std + grp.value_rms.mean
        else:
            vals_denorm = vals
            bootstrap_denorm = bootstrap

        # ── GAE ──────────────────────────────────────────────
        advantages, returns = self._compute_gae(
            rews, vals_denorm, terms, bootstrap_denorm,
            self.config.gamma, self.config.gae_lambda,
        )

        # ── normalise returns for critic targets ─────────────
        if self.config.use_value_normalization:
            grp.value_rms.update(returns)
            returns_norm = (returns - grp.value_rms.mean) / grp.value_rms.std
        else:
            returns_norm = returns

        obs_t  = torch.stack(obs_list).to(self.device)
        acts_t = torch.tensor(act_list, dtype=torch.long, device=self.device)
        lps_t  = torch.tensor(lp_list, dtype=torch.float32, device=self.device)
        adv_t  = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t  = torch.tensor(returns_norm, dtype=torch.float32, device=self.device)

        return obs_t, acts_t, lps_t, adv_t, ret_t

    @staticmethod
    def _compute_gae(rewards, values, terminateds, bootstrap, gamma, gae_lam):
        """
        GAE-λ with episode-boundary masking.
        terminateds[t] = 1 means the environment truly ended after step t.
        """
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        next_vals = np.append(values[1:], bootstrap)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - terminateds[t]
            delta = rewards[t] + gamma * next_vals[t] * mask - values[t]
            gae = delta + gamma * gae_lam * mask * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    # ──────────────────────────────────────────────────────────
    # PPO optimisation
    # ──────────────────────────────────────────────────────────

    def _ppo_epochs(self, grp, obs, acts, lps_old, adv, ret) -> dict:
        cfg = self.config
        N = obs.shape[0]
        metrics = defaultdict(float)
        n_iters = 0

        for epoch in range(cfg.ppo_epochs):
            if cfg.use_mini_batch and N > cfg.mini_batch_size:
                idx = torch.randperm(N, device=self.device)
                for start in range(0, N, cfg.mini_batch_size):
                    mb = idx[start:start + cfg.mini_batch_size]
                    m = self._ppo_step(grp, obs[mb], acts[mb], lps_old[mb],
                                       adv[mb], ret[mb])
                    for k, v in m.items():
                        metrics[k] += v
                    n_iters += 1
            else:
                m = self._ppo_step(grp, obs, acts, lps_old, adv, ret)
                for k, v in m.items():
                    metrics[k] += v
                n_iters += 1
                # Early stop if policy changed too much
                if m["approx_kl"] > cfg.target_kl:
                    break

        return {k: v / max(n_iters, 1) for k, v in metrics.items()}

    def _ppo_step(self, grp, obs, acts, lps_old, adv, ret) -> dict:
        cfg = self.config

        # ── Actor ────────────────────────────────────────────
        lps_new, entropy = grp.actor.evaluate_actions(obs, acts)
        ratio = torch.exp(lps_new - lps_old)
        s1 = ratio * adv
        s2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * adv
        actor_loss = -torch.min(s1, s2).mean()
        entropy_mean = entropy.mean()
        total_actor_loss = actor_loss - cfg.entropy_coef * entropy_mean

        grp.actor_opt.zero_grad()
        total_actor_loss.backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(grp.actor.parameters(), cfg.max_grad_norm)
        grp.actor_opt.step()

        # ── Critic ───────────────────────────────────────────
        vals_pred = grp.critic(obs)
        critic_loss = F.mse_loss(vals_pred, ret)

        grp.critic_opt.zero_grad()
        (cfg.value_loss_coef * critic_loss).backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(grp.critic.parameters(), cfg.max_grad_norm)
        grp.critic_opt.step()

        approx_kl = (lps_old - lps_new).mean().item()
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_mean.item(),
            "approx_kl": approx_kl,
        }

    # ──────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for name, grp in self.role_groups.items():
            torch.save({
                "actor": grp.actor.state_dict(),
                "critic": grp.critic.state_dict(),
                "actor_opt": grp.actor_opt.state_dict(),
                "critic_opt": grp.critic_opt.state_dict(),
                "value_rms_mean": grp.value_rms.mean,
                "value_rms_var": grp.value_rms.var,
                "value_rms_count": grp.value_rms.count,
            }, os.path.join(path, f"{name}.pt"))
        torch.save({
            "adv_rms_mean": self.adv_reward_rms.mean,
            "adv_rms_var": self.adv_reward_rms.var,
            "adv_rms_count": self.adv_reward_rms.count,
            "good_rms_mean": self.good_reward_rms.mean,
            "good_rms_var": self.good_reward_rms.var,
            "good_rms_count": self.good_reward_rms.count,
        }, os.path.join(path, "reward_rms.pt"))

    def load(self, path: str):
        for name, grp in self.role_groups.items():
            fp = os.path.join(path, f"{name}.pt")
            if not os.path.exists(fp):
                print(f"[WARN] checkpoint not found: {fp}")
                continue
            ckpt = torch.load(fp, map_location=self.device, weights_only=False)
            grp.actor.load_state_dict(ckpt["actor"])
            grp.critic.load_state_dict(ckpt["critic"])
            if "actor_opt" in ckpt:
                grp.actor_opt.load_state_dict(ckpt["actor_opt"])
            if "critic_opt" in ckpt:
                grp.critic_opt.load_state_dict(ckpt["critic_opt"])
            grp.value_rms.mean = ckpt.get("value_rms_mean", 0.0)
            grp.value_rms.var = ckpt.get("value_rms_var", 1.0)
            grp.value_rms.count = ckpt.get("value_rms_count", 1e-4)

        rp = os.path.join(path, "reward_rms.pt")
        if os.path.exists(rp):
            rms = torch.load(rp, map_location="cpu", weights_only=False)
            self.adv_reward_rms.mean = rms.get("adv_rms_mean", 0.0)
            self.adv_reward_rms.var = rms.get("adv_rms_var", 1.0)
            self.adv_reward_rms.count = rms.get("adv_rms_count", 1e-4)
            self.good_reward_rms.mean = rms.get("good_rms_mean", 0.0)
            self.good_reward_rms.var = rms.get("good_rms_var", 1.0)
            self.good_reward_rms.count = rms.get("good_rms_count", 1e-4)
