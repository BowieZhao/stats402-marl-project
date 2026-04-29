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
    max_cycles: int = 25           # mpe2 default (TUNED DOWN from 50; time pressure)
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
    plan_bias_strength: float = 2.0    # moderate bias; comm provides direction hint
                                        # (was 4.0 too strong, 0 broke actor entirely)

    # ═══════════════════════════════════════════════════════════
    # Reward parameters (final design)
    # ═══════════════════════════════════════════════════════════
    use_structured_reward: bool = True       # turn on/off the new reward stack
    R_distance_coef: float = -0.05           # weak distance shaping (toward UNCAUGHT good)
    R_capture: float = 12.0                  # collision with UNCAUGHT good (TUNED from 10)
    R_encircle_radius: float = 0.4           # radius for encircle group / pin_assist
    R_encircle_min_count: int = 3            # min advs in radius to trigger encircle
    R_encircle_per_extra: float = 2.0        # bonus per extra teammate
                                              # n=3→+4, n=4→+6
    R_progress_first: float = 30.0           # killer of 1st-caught good
    R_progress_second: float = 40.0          # killer of 2nd-caught good (escalation)
    R_pin_assist: float = 10.0               # encircle group at first-catch (not killer)
    R_team_bonus: float = 5.0                # all advs on any first-catch (light celebration)
    R_role_align: float = 0.5                # action matches plan_bias (moderate)
                                              # was 1.0 (E1 unfair), 0 (no signal at all)
    R_message_thrash_penalty: float = -3.0   # leader-only: unjustified message switch
    R_correct_message: float = 2.0           # small teaching signal for leader
    use_correct_message_reward: bool = True

    collision_distance: float = 0.12         # adv collide with good (mpe2 default)

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
