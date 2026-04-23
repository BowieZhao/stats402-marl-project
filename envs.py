"""
Unified wrapper around simple_world_comm_v3.parallel_env.

Key design choices:
  1. `agents` property delegates to inner env (dynamic — agents disappear when done)
  2. `possible_agents` is static and available before reset
  3. `observation_spaces` / `action_spaces` exposed for algorithm construction
  4. Optional comm ablation: zero out leader_comm in normal adversary obs
  5. No injection of omniscient information anywhere
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

try:
    from mpe2 import simple_world_comm_v3
except ImportError as e:
    raise ImportError(
        "Failed to import simple_world_comm_v3 from mpe2. "
        "Install: pip install mpe2"
    ) from e


# ─────────────────────────────────────────────────────────────
# Role helpers
# ─────────────────────────────────────────────────────────────

def is_leader(name: str) -> bool:
    return "leadadversary" in name

def is_adversary(name: str) -> bool:
    """True for ALL adversary-side agents (leader included)."""
    return "adversary" in name

def is_normal_adversary(name: str) -> bool:
    """True for non-leader adversaries only."""
    return "adversary" in name and "leadadversary" not in name

def is_good(name: str) -> bool:
    return name.startswith("agent_")

def get_role(name: str) -> str:
    if is_leader(name):
        return "leader"
    if is_normal_adversary(name):
        return "adversary"
    if is_good(name):
        return "good"
    raise ValueError(f"Cannot infer role from: {name}")


# ─────────────────────────────────────────────────────────────
# Agent spec dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class AgentSpec:
    name: str
    role: str          # "leader" / "adversary" / "good"
    obs_dim: int
    action_dim: int
    action_type: str   # "discrete" / "continuous"


# ─────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────

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
        self._specs_built = False

    # ─────────── Properties that delegate to inner env ───────────

    @property
    def possible_agents(self) -> List[str]:
        """Static list of all agents that could ever exist (available before reset)."""
        return list(self._env.possible_agents)

    @property
    def agents(self) -> List[str]:
        """Dynamic list of currently alive agents (shrinks as agents are done)."""
        return list(self._env.agents)

    @property
    def observation_spaces(self) -> Dict[str, Any]:
        return {a: self._env.observation_space(a) for a in self._env.possible_agents}

    @property
    def action_spaces(self) -> Dict[str, Any]:
        return {a: self._env.action_space(a) for a in self._env.possible_agents}

    # ─────────── Core API ────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        obs_dict, info_dict = self._env.reset(seed=seed)

        if not self._specs_built:
            self._build_specs(obs_dict)
            self._specs_built = True

        obs_dict = self._postprocess_obs(obs_dict)
        return obs_dict, info_dict

    def step(self, action_dict: Dict[str, Any]):
        next_obs, rewards, terminations, truncations, infos = self._env.step(action_dict)
        next_obs = self._postprocess_obs(next_obs)

        # Apply cooperative reward shaping if enabled
        if getattr(self.config, "use_coop_reward", False):
            rewards = self._reshape_rewards(rewards)

        return next_obs, rewards, terminations, truncations, infos

    def close(self):
        self._env.close()

    # ─────────── Agent spec helpers ──────────────────────────────

    def get_agent_spec(self, name: str) -> AgentSpec:
        return self.agent_specs[name]

    def get_obs_dim(self, name: str) -> int:
        return self.agent_specs[name].obs_dim

    def get_action_dim(self, name: str) -> int:
        return self.agent_specs[name].action_dim

    # ─────────── Internal ────────────────────────────────────────

    def _build_specs(self, obs_dict: Dict[str, np.ndarray]):
        for agent in self._env.possible_agents:
            obs = np.asarray(obs_dict[agent], dtype=np.float32)
            act_space = self._env.action_space(agent)

            if hasattr(act_space, "n"):
                action_type = "discrete"
                action_dim = int(act_space.n)
            else:
                action_type = "continuous"
                action_dim = int(np.prod(act_space.shape))

            self.agent_specs[agent] = AgentSpec(
                name=agent,
                role=get_role(agent),
                obs_dim=int(obs.shape[0]),
                action_dim=action_dim,
                action_type=action_type,
            )

    def _postprocess_obs(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Cast to float32.  Apply ablation masking for normal adversaries
        based on config.ablation_mode:
          "full"          → no masking
          "no_comm"       → zero out leader_comm (last 4 dims)
          "no_leader_pos" → zero out leader's rel position + velocity, keep comm
          "blind"         → zero out both leader position AND comm
        """
        mode = self.config.ablation_mode
        out = {}
        for agent, obs in obs_dict.items():
            obs = np.asarray(obs, dtype=np.float32)
            if is_normal_adversary(agent) and mode != "full":
                obs = obs.copy()
                if mode in ("no_comm", "blind"):
                    # Zero leader_comm: last comm_dim dims
                    obs[-self.config.comm_dim:] = 0.0
                if mode in ("no_leader_pos", "blind"):
                    # Zero leader's relative position
                    p0, p1 = self.config.leader_pos_dims
                    obs[p0:p1] = 0.0
                    # Zero leader's velocity
                    v0, v1 = self.config.leader_vel_dims
                    obs[v0:v1] = 0.0
            out[agent] = obs
        return out

    # ─────────── Reward reshaping ────────────────────────────────

    def _get_agent_positions(self) -> Dict[str, np.ndarray]:
        """
        Read absolute positions directly from the underlying MPE world.
        Used for reward reshaping.
        """
        positions = {}
        for a in self._env.unwrapped.world.agents:
            positions[a.name] = np.asarray(a.state.p_pos, dtype=np.float32)
        return positions

    def _reshape_rewards(self, raw_rewards: Dict[str, float]) -> Dict[str, float]:
        """
        Add cooperative bonuses on top of the raw MPE reward.

        The raw MPE adversary reward is already team-like but has weird structure:
          rew[i] = -0.1 * min_dist(i to any good) + 5 * (# ag-adv collisions world-wide)
        This means the collision term is shared across adversaries regardless of
        who caused it. We keep that (it's harmless), but ADD:

          (1) per-good "surround bonus" — if >= 2 adversaries are within
              `coop_radius` of the same good agent, each adversary on the scene
              gets `coop_bonus_per_extra` per additional teammate present.
              So 2 adv around a good = +coop_bonus_per_extra each,
                 3 adv around a good = +2*coop_bonus_per_extra each, etc.

          (2) "coverage bonus" — if EVERY good agent has at least one adversary
              within `coop_radius`, the entire adversary team gets `coverage_bonus`.
              Prevents the "all adversaries dogpile on one good" failure mode.

        Good agent rewards are untouched.
        """
        cfg = self.config
        radius = cfg.coop_radius
        bonus_per_extra = cfg.coop_bonus_per_extra
        coverage_bonus = cfg.coverage_bonus

        positions = self._get_agent_positions()

        # Identify adversary-side and good-side agents by world object
        adv_names = [a.name for a in self._env.unwrapped.world.agents if a.adversary]
        good_names = [a.name for a in self._env.unwrapped.world.agents if not a.adversary]

        if not adv_names or not good_names:
            return raw_rewards  # nothing to reshape

        # Per-adversary accumulator for the surround bonus
        adv_bonus = {a: 0.0 for a in adv_names}

        # For each good agent, count how many adversaries are within radius
        all_goods_covered = True
        for g in good_names:
            if g not in positions:
                all_goods_covered = False
                continue
            g_pos = positions[g]

            nearby_advs = []
            for a in adv_names:
                if a not in positions:
                    continue
                d = float(np.linalg.norm(positions[a] - g_pos))
                if d < radius:
                    nearby_advs.append(a)

            if len(nearby_advs) == 0:
                all_goods_covered = False

            # Surround bonus: each extra teammate beyond the first gives bonus
            # For n adversaries around a good, each of them gets (n - 1) * bonus_per_extra
            n = len(nearby_advs)
            if n >= 2:
                extra = (n - 1) * bonus_per_extra
                for a in nearby_advs:
                    adv_bonus[a] += extra

        # Coverage bonus applied to all adversaries
        team_coverage = coverage_bonus if all_goods_covered else 0.0

        # Apply
        out = dict(raw_rewards)
        for a in adv_names:
            if a in out:
                out[a] = float(out[a]) + adv_bonus[a] + team_coverage

        return out
