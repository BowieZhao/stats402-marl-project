"""
Configuration for MARL communication experiments.
Final design with explicit ABCD semantics, plan_bias, dynamic alpha, and structured reward.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════
    # Algorithm & experiment identity
    # ═══════════════════════════════════════════════════════════
    algo: str = "mappo"            # "mappo" only for new design (DQN/DDPG/PSO are baselines)
    exp_name: str = "exp"
    seed: int = 42
    device: str = "cpu"
    run_name: str = ""             # prefix for outputs

    # ═══════════════════════════════════════════════════════════
    # Environment
    # ═══════════════════════════════════════════════════════════
    num_good: int = 2
    num_adversaries: int = 4       # 1 leader + 3 normal
    num_forests: int = 2
    num_food: int = 2
    num_obstacles: int = 1
    max_cycles: int = 50          # back to 50 (25 starves the env of time)
    continuous_actions: bool = False
    render_mode: Optional[str] = None

    # ═══════════════════════════════════════════════════════════
    # Experiment condition (E1 / E2 / E3)
    # ═══════════════════════════════════════════════════════════
    # condition controls: message availability + plan_bias + alpha learning
    #   "E1_full"       : full method (message learned, plan_bias on, alpha learnable)
    #   "E2_no_comm"    : message blocked, plan_bias off, alpha N/A
    #   "E3_no_alpha"   : message learned, plan_bias on, alpha fixed at 0.5
    condition: str = "E1_full"

    # ═══════════════════════════════════════════════════════════
    # Communication / message
    # ═══════════════════════════════════════════════════════════
    comm_dim: int = 4              # A/B/C/D
    use_explicit_abcd: bool = True # Always True in new design

    # ═══════════════════════════════════════════════════════════
    # Plan_bias (rule-based message decoder)
    # ═══════════════════════════════════════════════════════════
    plan_bias_threshold: float = 0.05  # ignore very small position deltas
    plan_bias_strength: float = 1.0    # weakened from 2.0: hint, not command

    # ═══════════════════════════════════════════════════════════
    # Reward parameters — final clean design.
    #
    # Capture is COLLECTIVE: a good is caught iff, in a single step,
    # ≥ R_encircle_min_count advs are within R_encircle_radius AND
    # at least one adv is within collision_distance. Once caught, the good
    # disappears for all reward purposes and is masked from adv obs.
    #
    # Episode terminates as soon as ALL goods are caught (early termination).
    #
    # Five real-task rewards. No teaching rewards.
    # ═══════════════════════════════════════════════════════════
    use_structured_reward: bool = True

    # ─── Real-task rewards ───
    R_distance_coef: float = -0.05           # weak distance shaping toward nearest UNCAUGHT good
    R_encircle_radius: float = 0.4
    R_encircle_min_count: int = 3
    R_encircle_per_extra: float = 4.0        # n=3 → +8/adv, n=4 → +12/adv
    R_progress_first: float = 30.0           # team-wide on 1st good caught
    R_progress_second: float = 50.0          # team-wide on 2nd good caught
    R_no_catch_penalty: float = -20.0        # per adv at terminal step if 0 catches all episode

    # ─── Speed bonus ───
    # One-shot team-wide bonus on the step that catches the LAST good
    # (i.e., the step that triggers early termination). Scales linearly
    # with steps remaining: catching at step 10 with max_cycles=50 gives
    # +(50-10-1)*coef = +39 per adv. Caps at ≈ R_progress_second's order
    # of magnitude so it doesn't dominate.
    R_speed_bonus_coef: float = 1.0          # per remaining step, per adv

    # ─── Teaching rewards: ALL DISABLED ───
    R_capture: float = 0.0                   # capture is now a collective event,
                                              # not a per-adv collision reward
    R_role_align: float = 0.0
    R_correct_message_per_step: float = 0.0
    R_wrong_message_per_step: float = 0.0
    R_stale_message_penalty: float = 0.0

    # 【新增】Follower 执行了 Leader 期望动作的局部奖励 (Gain = 1.0)
    R_action_matches_plan_bias: float = 0.0
    # 【修改】Leader 无端切换信号的惩罚 (Gain = -3.0)
    R_message_thrash_penalty: float = -3.0

    # ─── Deprecated (kept as 0 for backward compat) ───
    R_progress: float = 30.0                 # legacy alias for R_progress_first
    R_progress_team_share: float = 0.0       # DISABLED — superseded by team-wide R_progress
    R_pin_assist: float = 0.0                # DISABLED
    R_team_bonus: float = 0.0                # DISABLED
    R_correct_message: float = 0.0           # DISABLED — replaced by per-step dense version
    use_correct_message_reward: bool = False # DISABLED — using new dense per-step logic

    collision_distance: float = 0.12         # mpe2 default

    # ═══════════════════════════════════════════════════════════
    # Adversary architecture (alpha-gated)
    # ═══════════════════════════════════════════════════════════
    alpha_init: float = 0.5
    alpha_learnable: bool = True   # E1/E2: True; E3: False (fixed at 0.5)

    # ═══════════════════════════════════════════════════════════
    # Obs dimensions (computed automatically based on flags)
    # ═══════════════════════════════════════════════════════════
    # Leader:    34 + 4 (last_msg) + 4 (forest_flags) = 42
    # Adversary: 34 + 4 (message_one_hot) = 38
    # These are HANDLED in envs.py (post-processing) — agents see expanded obs.

    # ═══════════════════════════════════════════════════════════
    # Frozen good policy
    # ═══════════════════════════════════════════════════════════
    frozen_good_path: Optional[str] = None
    frozen_good_deterministic: bool = False

    # ═══════════════════════════════════════════════════════════
    # PPO training hyperparameters
    # ═══════════════════════════════════════════════════════════
    total_episodes: int = 4000
    update_every: int = 8           # episodes between PPO updates
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    ppo_epochs: int = 4
    minibatch_size: int = 256

    # ═══════════════════════════════════════════════════════════
    # Logging / checkpointing
    # ═══════════════════════════════════════════════════════════
    log_every: int = 20
    eval_every: int = 100
    eval_episodes: int = 10
    save_every: int = 500
    log_dir: str = "outputs/logs"
    model_dir: str = "outputs/models"

    # ═══════════════════════════════════════════════════════════
    # Legacy fields kept for envs.py backward compat
    # (still used by ablation_mode logic if condition is overridden)
    # ═══════════════════════════════════════════════════════════
    ablation_mode: str = "full"    # mostly unused in new design
    use_coop_reward: bool = True   # always True (new reward is coop)


    # ═══════════════════════════════════════════════════════════
    # Helper: derive condition flags
    # ═══════════════════════════════════════════════════════════
    @property
    def message_enabled(self) -> bool:
        """True if leader message reaches normal advs."""
        return self.condition in ("E1_full", "E3_no_alpha")

    @property
    def plan_bias_enabled(self) -> bool:
        """True if plan_bias is computed and applied to adv logits."""
        return self.condition in ("E1_full", "E3_no_alpha")

    @property
    def alpha_is_learnable(self) -> bool:
        """True if alpha is learned (False = fixed at alpha_init)."""
        if self.condition == "E2_no_comm":
            return False  # alpha doesn't matter
        if self.condition == "E3_no_alpha":
            return False  # alpha fixed
        return self.alpha_learnable  # default True for E1