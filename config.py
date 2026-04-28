"""
Central configuration for simple_world_comm_v3 MARL experiments.
All hyperparameters live here; main.py can override via CLI.
"""

import os
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch


@dataclass
class Config:
    # ──────────────────────────────────────────────
    # Algorithm selection
    # ──────────────────────────────────────────────
    algo: str = "mappo"

    # ──────────────────────────────────────────────
    # Environment  (simple_world_comm_v3 defaults)
    # ──────────────────────────────────────────────
    env_name: str = "simple_world_comm_v3"
    num_good: int = 2
    num_adversaries: int = 4        # includes leader (1 leader + 3 normal)
    num_obstacles: int = 1
    num_food: int = 2
    num_forests: int = 2           # per professor: keep default 2
    max_cycles: int = 50            # 25 too short; 100 inflates variance
    continuous_actions: bool = False
    dynamic_rescaling: bool = False
    render_mode: Optional[str] = None

    # ──────────────────────────────────────────────
    # Ablation mode (key experimental variable)
    # ──────────────────────────────────────────────
    # "full"          : normal adversaries see everything (leader pos + comm)
    # "no_comm"       : zero out leader_comm dims only (leader pos still visible)
    # "no_leader_pos" : zero out leader's relative position + velocity, keep comm
    # "blind"         : zero out both leader position AND comm
    #
    # The critical finding: "no_comm" ≈ "full" because adversaries can follow
    # the leader's movement trajectory (implicit communication).
    # "no_leader_pos" vs "blind" isolates the true value of explicit comm.
    ablation_mode: str = "full"
    comm_dim: int = 4               # leader_comm one-hot at obs[-4:]
    leader_pos_dims: tuple = (14, 16)   # leader rel position in normal adv obs
    leader_vel_dims: tuple = (24, 26)   # leader velocity in normal adv obs

    # ──────────────────────────────────────────────
    # Cooperative reward shaping
    # ──────────────────────────────────────────────
    # The raw MPE adversary reward treats all adversaries symmetrically
    # (every adv gets +5 for any world-wide ag-adv collision). This makes
    # cooperation vs independence indistinguishable in reward.
    #
    # We add two bonuses on top of the raw reward (only when enabled):
    #   1. Surround bonus: if 2+ adversaries are within coop_radius of the
    #      same good agent, each of them gets (n-1) * coop_bonus_per_extra.
    #      This rewards joint pursuit of the same target.
    #   2. Coverage bonus: if every good agent has at least one adversary
    #      within coop_radius, the whole adversary team gets coverage_bonus.
    #      This prevents the "everyone dogpile on one good" failure mode
    #      and forces division of labor.
    #
    # With both together, the optimal strategy requires coordinated target
    # assignment -- exactly what leader communication should enable.
    use_coop_reward: bool = False
    coop_radius: float = 0.5         # distance threshold for "near" a good
                                     # 0.3 is too strict (random policy never triggers)
                                     # 0.5 is ~25% of field; trainable policies trigger
    coop_bonus_per_extra: float = 2.0  # per extra teammate around a good, per step
    coverage_bonus: float = 3.0       # when ALL goods have an adv nearby, per step

    # ──────────────────────────────────────────────
    # Frozen good policy (for fair cross-algorithm comparison)
    # ──────────────────────────────────────────────
    # When set, the good agents use a pre-trained, frozen policy loaded from
    # this checkpoint path, rather than learning alongside the adversaries.
    # This is used in Phase 2+3 so that all adversary-side algorithms
    # (MAPPO variants, DQN, DDPG, PSO) face an identical opponent.
    frozen_good_path: Optional[str] = None
    frozen_good_deterministic: bool = False  # True = argmax, False = sample

    # ──────────────────────────────────────────────
    # Training schedule
    # ──────────────────────────────────────────────
    train_team: str = "all"         # "all" = train both adversary + good

    total_episodes: int = 4000     # converges around 2500-3000; 4000 gives margin
    # Collect this many episodes before each PPO update.
    # 8 eps × 50 steps × 3 agents/group ≈ 1200 transitions per group.
    update_every_n_episodes: int = 8

    eval_every_n_episodes: int = 100   # more frequent eval for cleaner curves
    eval_episodes: int = 10
    save_every_n_episodes: int = 500
    print_every: int = 20

    # ──────────────────────────────────────────────
    # PPO hyperparameters
    # ──────────────────────────────────────────────
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.15        # slightly tighter than default 0.2
    ppo_epochs: int = 4
    use_mini_batch: bool = False    # full-batch by default; enable for large batches
    mini_batch_size: int = 128

    actor_lr: float = 3e-4
    critic_lr: float = 5e-4         # critic can learn faster
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.02      # encourage exploration
    value_loss_coef: float = 0.5
    target_kl: float = 0.15         # early-stop PPO epochs if KL exceeds this

    # ──────────────────────────────────────────────
    # Network architecture
    # ──────────────────────────────────────────────
    hidden_dim: int = 128

    # ──────────────────────────────────────────────
    # Reward processing
    # ──────────────────────────────────────────────
    # Divide each step reward by its side's running std.
    # Most impactful single stabilisation for spike-heavy envs.
    use_reward_normalization: bool = True

    # Optional bounded squeeze after normalisation.
    use_reward_tanh: bool = False
    reward_tanh_scale: float = 5.0

    # ──────────────────────────────────────────────
    # Value normalisation
    # ──────────────────────────────────────────────
    # Normalise critic targets (returns) via per-role running mean/std.
    # Helps the critic when reward scale drifts during training.
    use_value_normalization: bool = True

    # ──────────────────────────────────────────────
    # Logging & I/O
    # ──────────────────────────────────────────────
    log_dir: str = "outputs/logs"
    model_dir: str = "outputs/models"
    run_name: str = "swc_mappo"

    # ──────────────────────────────────────────────
    # Misc
    # ──────────────────────────────────────────────
    device: str = "cpu"
    seed: int = 42

    # ──────────────────────────────────────────────
    # Derived (auto-generated)
    # ──────────────────────────────────────────────
    @property
    def exp_name(self) -> str:
        tag = f"{self.algo}_{self.ablation_mode}"
        if self.use_coop_reward:
            tag += "_coop"
        if self.frozen_good_path:
            tag += "_fg"  # frozen good
        if self.num_good != 2:
            tag += f"_g{self.num_good}"
        return f"{self.run_name}_{tag}_s{self.seed}"

    # ──────────────────────────────────────────────
    # Utility methods
    # ──────────────────────────────────────────────
    def make_dirs(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def set_seed_everywhere(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def save_json(self, path: str):
        d = {}
        for k, v in asdict(self).items():
            d[k] = v
        d["exp_name"] = self.exp_name  # add derived field
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=str)
