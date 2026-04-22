from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

try:
    from mpe2 import simple_world_comm_v3
except ImportError as e:
    raise ImportError("pip install mpe2") from e


# =========================================================
# ROLE HELPERS
# =========================================================
def is_leader(name: str) -> bool:
    return "leadadversary" in name

def is_normal_adversary(name: str) -> bool:
    return "adversary" in name and "leadadversary" not in name

def is_good(name: str) -> bool:
    return name.startswith("agent_")


# =========================================================
# SPEC
# =========================================================
@dataclass
class AgentSpec:
    name: str
    role: str
    obs_dim: int
    action_dim: int
    action_type: str


# =========================================================
# ENV WRAPPER (STABLE VERSION)
# =========================================================
class WorldCommEnv:

    def __init__(self, config):
        self.config = config

        self._env = simple_world_comm_v3.parallel_env(
            num_good=config.num_good,
            num_adversaries=config.num_adversaries,
            num_obstacles=config.num_obstacles,
            num_food=config.num_food,
            max_cycles=config.max_cycles,
            num_forests=config.num_forests,
            continuous_actions=config.continuous_actions,
            render_mode=config.render_mode,
            dynamic_rescaling=config.dynamic_rescaling,
        )

        self.agent_specs: Dict[str, AgentSpec] = {}
        self._built = False

        # =====================================================
        # PER-AGENT NORMALIZATION (FIX CRITICAL BUG)
        # =====================================================
        self.obs_mean: Dict[str, np.ndarray] = {}
        self.obs_var: Dict[str, np.ndarray] = {}
        self.obs_count: Dict[str, float] = {}

    # =========================================================
    # PROPERTIES
    # =========================================================
    @property
    def possible_agents(self):
        return list(self._env.possible_agents)

    @property
    def agents(self):
        return list(self._env.agents)

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, seed: Optional[int] = None):
        obs, info = self._env.reset(seed=seed)

        if not self._built:
            self._build_specs(obs)
            self._built = True

        obs = self._process_obs(obs, update_stats=True)

        return obs, info

    # =========================================================
    # STEP
    # =========================================================
    def step(self, actions):
        obs, rewards, terms, truncs, infos = self._env.step(actions)

        obs = self._process_obs(obs, update_stats=False)
        rewards = self._process_rewards(rewards)

        return obs, rewards, terms, truncs, infos

    def close(self):
        self._env.close()

    # =========================================================
    # OBS PROCESSING (SAFE)
    # =========================================================
    def _process_obs(self, obs_dict, update_stats=False):

        out = {}

        for agent, obs in obs_dict.items():

            obs = np.asarray(obs, dtype=np.float32)

            # -----------------------------
            # SOFT ABLATION (IMPORTANT FIX)
            # -----------------------------
            if is_normal_adversary(agent) and self.config.ablation_mode != "full":

                if self.config.ablation_mode in ("no_comm", "blind"):
                    obs[-self.config.comm_dim:] *= 0.1

                if self.config.ablation_mode in ("no_leader_pos", "blind"):
                    p0, p1 = self.config.leader_pos_dims
                    v0, v1 = self.config.leader_vel_dims

                    obs[p0:p1] *= 0.1
                    obs[v0:v1] *= 0.1

            # -----------------------------
            # CLIP (CRITICAL)
            # -----------------------------
            obs = np.clip(obs, -10, 10)

            # -----------------------------
            # UPDATE STATS
            # -----------------------------
            if update_stats:
                self._update_stats(agent, obs)

            # -----------------------------
            # NORMALIZE
            # -----------------------------
            obs = self._normalize(agent, obs)

            out[agent] = obs

        return out

    # =========================================================
    # REWARD PROCESSING
    # =========================================================
    def _process_rewards(self, rewards):

        out = {}

        for k, v in rewards.items():

            v = float(v)

            # clip reward (PPO stability)
            v = np.clip(v, -10, 10)

            # scale down
            v *= 0.1

            out[k] = v

        return out

    # =========================================================
    # NORMALIZATION (FIX BROADCAST ERROR)
    # =========================================================
    def _update_stats(self, agent, obs):

        obs = obs.reshape(-1)

        if agent not in self.obs_mean:
            self.obs_mean[agent] = np.zeros_like(obs)
            self.obs_var[agent] = np.ones_like(obs)
            self.obs_count[agent] = 1e-8

        self.obs_count[agent] += 1

        delta = obs - self.obs_mean[agent]

        self.obs_mean[agent] += delta / self.obs_count[agent]
        self.obs_var[agent] += delta * (obs - self.obs_mean[agent])

    def _normalize(self, agent, obs):

        if agent not in self.obs_mean:
            return obs

        mean = self.obs_mean[agent]
        var = self.obs_var[agent]
        count = self.obs_count[agent]

        std = np.sqrt(var / count + 1e-8)

        return (obs - mean) / (std + 1e-8)

    # =========================================================
    # BUILD SPEC
    # =========================================================
    def _build_specs(self, obs_dict):

        for agent, obs in obs_dict.items():

            obs = np.asarray(obs, dtype=np.float32)
            act = self._env.action_space(agent)

            if hasattr(act, "n"):
                action_type = "discrete"
                action_dim = int(act.n)
            else:
                action_type = "continuous"
                action_dim = int(np.prod(act.shape))

            self.agent_specs[agent] = AgentSpec(
                name=agent,
                role=self._role(agent),
                obs_dim=int(obs.shape[0]),
                action_dim=action_dim,
                action_type=action_type,
            )

    def _role(self, name):
        if is_leader(name):
            return "leader"
        if is_normal_adversary(name):
            return "adversary"
        if is_good(name):
            return "good"
        return "unknown"

    # =========================================================
    # API
    # =========================================================
    def get_agent_spec(self, name):
        return self.agent_specs[name]