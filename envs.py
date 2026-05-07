"""
Environment wrapper for MPE2 simple_world_comm with the FINAL design:
  - Explicit ABCD message semantics
  - Plan_bias rule-based decoder
  - Last-message + forest-flags injection into leader obs
  - Plan_bias injection into adv obs (for actor input only; the actual logit
    combination happens in the actor module).
  - Structured cooperative reward (encircle, capture, progress, role_align)
  - Message thrash penalty (anti-spam on leader)


═══════════════════════════════════════════════════════════════════
  EXPLICIT ABCD MESSAGE DEFINITIONS
═══════════════════════════════════════════════════════════════════

The leader's message space is {0, 1, 2, 3} = {A, B, C, D}.
Each message has a FIXED, GROUNDED semantic, defined via the plan_bias
function below. The leader does NOT learn what each message means — the
meanings are pre-defined by `compute_plan_bias()`. The leader only learns
WHEN to send each message based on the global state.

  Message ID │ Symbol │ Semantic                       │ Adv response
  ───────────┼────────┼────────────────────────────────┼──────────────
       0     │   A    │  "focus good_0"                │  Move toward good_0
       1     │   B    │  "focus good_1"                │  Move toward good_1
       2     │   C    │  "rush forest_0"               │  Move toward forest_0
       3     │   D    │  "rush forest_1"               │  Move toward forest_1

Rationale:
  - A/B let the leader concentrate the team on a specific exposed good.
  - C/D let the leader send the team into a forest where a good is hidden
    (only the leader can see goods inside forests; normal advs cannot).
  - Cooperation modes (encircle vs split) emerge via the alpha mechanism:
    if an adv is very close to the OTHER good than the one leader directed
    to, alpha learns to be high, and that adv stays to catch the closer one
    (split formation emerges).

When leader should switch:
  - Caught a good → switch to focus the other good (A↔B).
  - good entered a forest → switch from A/B to C/D.
  - good exited forest → switch from C/D back to A/B.
The R_message_thrash_penalty discourages switches that aren't justified by
either of the above conditions, preventing message "thrashing".
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from mpe2 import simple_world_comm_v3


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def is_leader(name: str) -> bool:
    return "leadadversary" in name


def is_normal_adversary(name: str) -> bool:
    return "adversary" in name and "leadadversary" not in name


def is_good(name: str) -> bool:
    return name.startswith("agent_")


@dataclass
class AgentSpec:
    name: str
    role: str          # "leader" / "adversary" / "good"
    obs_dim: int       # POST-augmentation dim (42 for leader, 38 for adv, 28 for good)
    action_dim: int
    action_type: str   # "discrete"


# ═══════════════════════════════════════════════════════════════
# WorldCommEnv: PettingZoo parallel API wrapper with all extensions
# ═══════════════════════════════════════════════════════════════
class WorldCommEnv:
    """
    Parallel-API MPE2 wrapper with full FINAL design extensions.
    """

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
        )

        # Initialize names
        self._env.reset(seed=config.seed)
        self.possible_agents = list(self._env.possible_agents)

        # Episode state tracking (set in reset())
        self._last_message: int = 0
        self._last_forest_state: Tuple[int, ...] = (0, 0, 0, 0)
        self._caught_set_this_step: set = set()
        self._goods_caught_ever: set = set()  # track per-episode first catches
        self._step_count: int = 0  # for speed bonus computation

        # Cache leader/adv/good name lists from world
        # (we'll populate after reset)
        self._leader_names = []
        self._normal_adv_names = []
        self._good_names = []

    # ─── PettingZoo-like API ─────────────────────────────────────
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            obs_dict, info = self._env.reset(seed=seed)
        else:
            obs_dict, info = self._env.reset()

        # Refresh agent name lists
        world_agents = self._env.unwrapped.world.agents
        self._leader_names = [a.name for a in world_agents
                              if a.adversary and a.leader]
        self._normal_adv_names = [a.name for a in world_agents
                                  if a.adversary and not a.leader]
        self._good_names = [a.name for a in world_agents if not a.adversary]

        # Reset episode tracking
        self._last_message = 0
        self._last_forest_state = self._compute_forest_state()
        self._caught_set_this_step = set()
        self._goods_caught_ever = set()
        self._step_count = 0
        self._eval_mode = False  # default: training mode (all rewards active)

        # Augment obs (post-process)
        obs_aug = self._augment_obs(obs_dict, current_message=0)
        return obs_aug, info

    def step(self, action_dict: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        # Extract message from leader's action BEFORE stepping
        leader_message = self._extract_leader_message(action_dict)

        # Step environment
        next_obs, rewards, terms, truncs, infos = self._env.step(action_dict)
        next_obs = {k: np.asarray(v, dtype=np.float32) for k, v in next_obs.items()}

        # Increment our own step counter (for speed bonus)
        self._step_count += 1

        # ── Compute structured rewards (replaces raw mpe2 rewards for advs) ──
        if self.config.use_structured_reward:
            self._caught_set_this_step = self._compute_caught_set()
            new_first_catches = self._caught_set_this_step - self._goods_caught_ever
            self._goods_caught_ever |= new_first_catches

            structured_rewards = self._compute_structured_rewards(
                action_dict=action_dict,
                leader_message=leader_message,
                new_first_catches=new_first_catches,
            )
            # Replace adversary-side rewards
            for k, v in structured_rewards.items():
                rewards[k] = float(v)
        else:
            self._caught_set_this_step = set()

        # ── Update tracking state for next step ──
        current_forest_state = self._compute_forest_state()
        self._last_forest_state = current_forest_state
        self._last_message = leader_message

        # ── Early termination: all goods caught → end episode now ──
        # We set TERMS (not truncs) because "all goods caught" is a true
        # terminal state — no further reward is achievable. GAE bootstrap
        # will correctly use V=0 for these states.
        all_caught = (len(self._goods_caught_ever) >= len(self._good_names)
                      and len(self._good_names) > 0)
        if all_caught:
            for k in list(terms.keys()):
                terms[k] = True

        # ── Terminal no-catch penalty (active in train AND eval) ──
        # If episode is ending AND we caught nothing all episode, penalize
        # all advs. This is a strong signal against "wandering" policies.
        any_terminating = any(terms.values()) or any(truncs.values())
        if (any_terminating
                and self.config.use_structured_reward
                and len(self._goods_caught_ever) == 0):
            adv_names = self._leader_names + self._normal_adv_names
            for a in adv_names:
                if a in rewards:
                    rewards[a] += self.config.R_no_catch_penalty

        # ── Augment obs ──
        obs_aug = self._augment_obs(next_obs, current_message=leader_message)

        return obs_aug, rewards, terms, truncs, infos

    def close(self):
        self._env.close()

    def set_eval_mode(self, eval_mode: bool):
        """When True, teaching rewards (role_align, correct_message, thrash)
        are excluded from reward computation. Used for clean EVAL metrics."""
        self._eval_mode = eval_mode

    # ─── Agent specs ────────────────────────────────────────────
    def get_agent_spec(self, name: str) -> AgentSpec:
        """Returns the augmented obs/action dim for the agent."""
        # Get raw obs space
        raw_obs_dim = int(np.prod(self._env.observation_space(name).shape))
        action_dim = int(self._env.action_space(name).n)

        if is_leader(name):
            # 34 + 4 (last_msg) + 4 (forest_flags) = 42
            obs_dim = raw_obs_dim + 4 + 4
            role = "leader"
        elif is_normal_adversary(name):
            # 34 + 4 (message_one_hot) = 38
            obs_dim = raw_obs_dim + 4
            role = "adversary"
        else:
            # Good agent: no augmentation
            obs_dim = raw_obs_dim
            role = "good"

        return AgentSpec(
            name=name, role=role,
            obs_dim=obs_dim, action_dim=action_dim,
            action_type="discrete",
        )

    # ─── Obs augmentation ───────────────────────────────────────
    def _augment_obs(self, obs_dict: Dict, current_message: int) -> Dict:
        """Apply per-role obs augmentation."""
        out = {}
        for name, obs in obs_dict.items():
            obs = np.asarray(obs, dtype=np.float32)

            if is_leader(name):
                # Append last_message one-hot (4) + forest_flags (4)
                last_msg_onehot = np.zeros(4, dtype=np.float32)
                last_msg_onehot[self._last_message] = 1.0

                forest_state = self._compute_forest_state()
                forest_flags = np.array(forest_state, dtype=np.float32)

                out[name] = np.concatenate([obs, last_msg_onehot, forest_flags])

            elif is_normal_adversary(name):
                # Mask already-caught goods from adv observation: their
                # rel_pos and vel are zeroed out so adv perceives them as
                # "gone" and won't drift toward / hover near them.
                # Layout (raw 34-dim adv obs):
                #   obs[20:22]  good_0 rel_pos
                #   obs[22:24]  good_1 rel_pos
                #   obs[24:26]  good_0 vel
                #   obs[26:28]  good_1 vel
                obs = obs.copy()
                if "agent_0" in self._goods_caught_ever and obs.shape[0] >= 28:
                    obs[20:22] = 0.0
                    obs[24:26] = 0.0
                if "agent_1" in self._goods_caught_ever and obs.shape[0] >= 28:
                    obs[22:24] = 0.0
                    obs[26:28] = 0.0

                # Append message one-hot (4) — the message leader broadcasted this step
                if self.config.message_enabled:
                    msg_onehot = np.zeros(4, dtype=np.float32)
                    msg_onehot[current_message] = 1.0
                else:
                    # E2_no_comm: message blocked
                    msg_onehot = np.zeros(4, dtype=np.float32)
                out[name] = np.concatenate([obs, msg_onehot])

            else:
                # Good agent: no change
                out[name] = obs

        return out

    # ─── Helpers: leader message extraction ─────────────────────
    def _compute_correct_message(self) -> int:
        """
        Compute the "correct" message leader should send given current state.
        Used by R_correct_message to teach leader the message → state mapping.

        Priority:
          1. If any good is in a forest, send C/D for that forest
             (forest takes priority because non-leader advs can't see goods there)
          2. If only one good is uncaught, focus on it (A or B)
          3. Otherwise, no specific 'correct' message — return 0 (A) as default
             (any of A/B is fine when both goods exposed)
        """
        forest_state = self._compute_forest_state()
        # forest_state = (g0_in_f0, g0_in_f1, g1_in_f0, g1_in_f1)

        # Rule 1: forest priority
        # Check forest_0 first (any good in forest_0)
        if forest_state[0] or forest_state[2]:
            # Need to send C, but only if the good in forest_0 is uncaught
            g0_in_f0 = forest_state[0] and ("agent_0" not in self._goods_caught_ever)
            g1_in_f0 = forest_state[2] and ("agent_1" not in self._goods_caught_ever)
            if g0_in_f0 or g1_in_f0:
                return 2  # C
        if forest_state[1] or forest_state[3]:
            g0_in_f1 = forest_state[1] and ("agent_0" not in self._goods_caught_ever)
            g1_in_f1 = forest_state[3] and ("agent_1" not in self._goods_caught_ever)
            if g0_in_f1 or g1_in_f1:
                return 3  # D

        # Rule 2: only one uncaught good → focus on it
        uncaught = [g for g in self._good_names
                    if g not in self._goods_caught_ever]
        if len(uncaught) == 1:
            # Force A or B based on which is left
            if uncaught[0].endswith("_0"):
                return 0  # A
            else:
                return 1  # B

        # Rule 3: both goods uncaught and exposed → A is fine (default)
        return 0  # A

    def _extract_leader_message(self, action_dict: Dict) -> int:
        """Decode leader's joint action into the message component.

        Leader's discrete action space is the Cartesian product
            [say_0, say_1, say_2, say_3] X [no_action, L, R, D, U]
        giving Discrete(20). The encoding is [say, move]:
            say_idx  = a // 5
            move_idx = a % 5
        """
        for ln in self._leader_names:
            if ln in action_dict:
                a = int(action_dict[ln])
                say_idx = a // 5
                return say_idx
        return 0

    # ─── Helpers: state computations ────────────────────────────
    def _get_agent_positions(self) -> Dict[str, np.ndarray]:
        positions = {}
        for a in self._env.unwrapped.world.agents:
            positions[a.name] = np.array(a.state.p_pos, dtype=np.float32)
        for lm in self._env.unwrapped.world.landmarks:
            positions[lm.name] = np.array(lm.state.p_pos, dtype=np.float32)
        return positions

    def _compute_forest_state(self) -> Tuple[int, int, int, int]:
        """
        Returns (g0_in_f0, g0_in_f1, g1_in_f0, g1_in_f1) as 0/1 ints.
        A good is "in" a forest if its distance to the forest center is
        less than the forest's size (radius).
        """
        positions = self._get_agent_positions()
        forests = [a for a in self._env.unwrapped.world.landmarks
                   if a.name.startswith("forest")]
        goods = sorted([a for a in self._env.unwrapped.world.agents
                        if not a.adversary], key=lambda x: x.name)

        if len(forests) < 2 or len(goods) < 2:
            # Not enough forests/goods for full flags; pad with 0
            return (0, 0, 0, 0)

        flags = []
        for g in goods[:2]:
            for f in forests[:2]:
                d = float(np.linalg.norm(g.state.p_pos - f.state.p_pos))
                in_forest = 1 if d < f.size else 0
                flags.append(in_forest)
        return tuple(flags)

    def _compute_caught_set(self) -> set:
        """
        A good is caught iff, IN A SINGLE STEP, both conditions hold:
          (i) ≥ R_encircle_min_count advs are within R_encircle_radius of it
          (ii) ≥ 1 adv is within collision_distance of it

        The colliding adv IS counted in (i) since collision_distance <
        R_encircle_radius. So a 3-adv catch is possible: 1 collider + 2
        outer ring. Once caught, the good is added to _goods_caught_ever
        and ignored by all subsequent reward + obs computation.
        """
        positions = self._get_agent_positions()
        adv_names = self._leader_names + self._normal_adv_names
        cfg = self.config

        caught = set()
        for g in self._good_names:
            if g not in positions:
                continue
            if g in self._goods_caught_ever:
                continue  # already gone; produces no further events

            n_in_encircle = 0
            any_collision = False
            for a in adv_names:
                if a not in positions:
                    continue
                d = float(np.linalg.norm(positions[a] - positions[g]))
                if d < cfg.R_encircle_radius:
                    n_in_encircle += 1
                if d < cfg.collision_distance:
                    any_collision = True

            if n_in_encircle >= cfg.R_encircle_min_count and any_collision:
                caught.add(g)
        return caught

    # ─── Plan_bias: rule-based message decoder ──────────────────
    def compute_plan_bias(self, message: int, adv_obs: np.ndarray) -> np.ndarray:
        """
        Translate message + adv's own obs into a 5-dim direction logit bias.
        Returns 5-dim numpy array [noop, L, R, D, U].

        adv_obs is the RAW (unaugmented) 34-dim adv obs.

        Layout (num_good=2, num_forests=2):
          obs[10:12]  forest_0 rel_pos
          obs[12:14]  forest_1 rel_pos
          obs[20:22]  good_0 rel_pos
          obs[22:24]  good_1 rel_pos
        """
        if not self.config.plan_bias_enabled:
            return np.zeros(5, dtype=np.float32)

        cfg = self.config
        # Take the unaugmented portion (first 34 dims)
        base = adv_obs[:34] if adv_obs.shape[0] > 34 else adv_obs

        if message == 0:    # A: focus good_0
            target = base[20:22]
        elif message == 1:  # B: focus good_1
            target = base[22:24]
        elif message == 2:  # C: rush forest_0
            target = base[10:12]
        elif message == 3:  # D: rush forest_1
            target = base[12:14]
        else:
            return np.zeros(5, dtype=np.float32)

        dx, dy = float(target[0]), float(target[1])
        bias = np.zeros(5, dtype=np.float32)  # [noop, L, R, D, U]
        thr = cfg.plan_bias_threshold
        s = cfg.plan_bias_strength

        if dx > thr:
            bias[2] = s   # right
        elif dx < -thr:
            bias[1] = s   # left

        if dy > thr:
            bias[4] = s   # up
        elif dy < -thr:
            bias[3] = s   # down

        return bias

    # ─── Structured reward computation ──────────────────────────
    def _compute_structured_rewards(
        self,
        action_dict: Dict,
        leader_message: int,
        new_first_catches: set,
    ) -> Dict[str, float]:
        """
        Final clean reward stack. NO teaching rewards.

          R_distance        -0.05 × dist(adv, nearest UNCAUGHT good), per step
          R_encircle        +(n-1)*4 to each adv in group, when n≥3 advs are
                            within 0.4 of an UNCAUGHT good
          R_progress_first  +30 team-wide on 1st good first-caught
          R_progress_second +50 team-wide on 2nd good first-caught
          R_speed_bonus     +1.0 × remaining_steps team-wide on the step
                            that completes ALL captures (one-shot)
          R_no_catch        -20 per adv at terminal step if 0 catches
                            (handled in step(), not here)

        Capture is a collective event: not a per-adv reward. The colliding
        adv gets the same R_progress as everyone else.
        """
        cfg = self.config
        positions = self._get_agent_positions()
        adv_names = self._leader_names + self._normal_adv_names
        good_names = self._good_names

        if not adv_names or not good_names:
            return {a: 0.0 for a in adv_names}

        rewards = {a: 0.0 for a in adv_names}

        # Snapshot: which goods were caught BEFORE this step's capture event.
        # Encircle and distance use this PRE snapshot — i.e. on the step where
        # a capture happens, that good is still treated as "uncaught" for
        # encircle/distance purposes (you get encircle credit for the
        # formation that produced the capture). Progress uses new_first_catches
        # directly so it fires exactly once per good.
        prev_goods_caught = self._goods_caught_ever - new_first_catches
        uncaught_goods = [g for g in good_names if g not in prev_goods_caught]

        # ── (1) R_distance: only toward UNCAUGHT goods ──
        for a in adv_names:
            if a not in positions:
                continue
            if not uncaught_goods:
                continue
            valid_dists = [np.linalg.norm(positions[a] - positions[g])
                           for g in uncaught_goods if g in positions]
            if valid_dists:
                min_d = float(min(valid_dists))
                rewards[a] += cfg.R_distance_coef * min_d

        # ── (2) R_encircle: only on UNCAUGHT goods ──
        for g in uncaught_goods:
            if g not in positions:
                continue
            g_pos = positions[g]
            in_group = []
            for a in adv_names:
                if a not in positions:
                    continue
                d = float(np.linalg.norm(positions[a] - g_pos))
                if d < cfg.R_encircle_radius:
                    in_group.append(a)
            n = len(in_group)
            if n >= cfg.R_encircle_min_count:
                bonus = cfg.R_encircle_per_extra * (n - 1)
                for a in in_group:
                    rewards[a] += bonus

        # ── (3) R_progress on first catches (TEAM-WIDE, escalating) ──
        for caught_g in new_first_catches:
            if caught_g not in positions:
                continue
            n_now_caught = len(self._goods_caught_ever)  # includes this catch
            if n_now_caught == 1:
                bonus = cfg.R_progress_first
            else:
                bonus = cfg.R_progress_second
            for a in adv_names:
                if a in rewards:
                    rewards[a] += bonus

        # ── (4) R_speed_bonus: one-shot on the step that catches the LAST good ──
        # Fires only when this step's catches push total caught up to num_good.
        # Bonus = coef × remaining_steps, given to every adv (team-wide).
        # Faster catch ⇒ more remaining steps ⇒ larger bonus.
        if (cfg.R_speed_bonus_coef > 0
                and len(new_first_catches) > 0
                and len(self._goods_caught_ever) >= cfg.num_good):
            remaining = max(0, cfg.max_cycles - self._step_count)
            speed_bonus = cfg.R_speed_bonus_coef * float(remaining)
            for a in adv_names:
                if a in rewards:
                    rewards[a] += speed_bonus

                # ── (5) 连入反馈环：R_action_matches_plan_bias (Follower听话奖励) ──
                # 逻辑：如果 Follower 当前动作与 plan_bias (指令) 推荐的方向一致，给予局部奖励
                if hasattr(cfg, 'R_action_matches_plan_bias') and cfg.R_action_matches_plan_bias > 0:
                    for a in self._normal_adv_names:
                        if a in action_dict and a in rewards:
                            adv_action = int(action_dict[a])
                            # 重构当前 Follower 的状态，作为解码器的输入
                            base_obs = self._reconstruct_adv_base_obs(a, positions)
                            # 计算前馈偏置 (plan_bias)
                            bias = self.compute_plan_bias(leader_message, base_obs)
                            # bias 是一个 5 维数组 [noop, L, R, D, U]。如果该动作上的 bias 大于 0，说明听话了
                            if bias[adv_action] > 0:
                                rewards[a] += cfg.R_action_matches_plan_bias

                # ── (6) 连入反馈环：R_message_thrash_penalty (Leader抗扰动惩罚) ──
                # 逻辑：如果 Leader 切换了指令，但环境状态(抓捕/进出树林)没变，则惩罚
                if hasattr(cfg, 'R_message_thrash_penalty') and cfg.R_message_thrash_penalty < 0:
                    if leader_message != getattr(self, '_last_message', leader_message):
                        # 判断切换是否有理：抓到新猎物了？或者树林状态变了？
                        justified = False
                        if len(new_first_catches) > 0:
                            justified = True
                        else:
                            current_forest_state = self._compute_forest_state()
                            last_f_state = getattr(self, '_last_forest_state', current_forest_state)
                            if current_forest_state != last_f_state:
                                justified = True

                        # 如果没有合理原因就切换，给所有 Leader 施加负增益
                        if not justified:
                            for ln in self._leader_names:
                                if ln in rewards:
                                    rewards[ln] += cfg.R_message_thrash_penalty


        return rewards

    def _any_uncaught_in_forest(self, forest_state: Tuple[int, int, int, int]) -> bool:
        """
        Check whether any UNCAUGHT good is currently in any forest.
        forest_state = (g0_in_f0, g0_in_f1, g1_in_f0, g1_in_f1).
        """
        g0_caught = "agent_0" in self._goods_caught_ever
        g1_caught = "agent_1" in self._goods_caught_ever
        # g0 in any forest and not caught
        if not g0_caught and (forest_state[0] or forest_state[1]):
            return True
        if not g1_caught and (forest_state[2] or forest_state[3]):
            return True
        return False

    def _reconstruct_adv_base_obs(self, adv_name: str,
                                    positions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Reconstruct the relevant slice of an adv's raw obs needed for plan_bias.
        We only need indices [10:14] (forest rel_pos) and [20:24] (good rel_pos).
        Build a 34-dim placeholder and fill those slices.
        """
        obs = np.zeros(34, dtype=np.float32)
        if adv_name not in positions:
            return obs
        adv_pos = positions[adv_name]

        # Forest rel positions
        forests = sorted([a for a in self._env.unwrapped.world.landmarks
                          if a.name.startswith("forest")], key=lambda x: x.name)
        for i, f in enumerate(forests[:2]):
            rel = f.state.p_pos - adv_pos
            obs[10 + 2*i:10 + 2*(i+1)] = rel

        # Good rel positions
        goods = sorted([a for a in self._env.unwrapped.world.agents
                        if not a.adversary], key=lambda x: x.name)
        for i, g in enumerate(goods[:2]):
            rel = g.state.p_pos - adv_pos
            obs[20 + 2*i:20 + 2*(i+1)] = rel

        return obs


# ═══════════════════════════════════════════════════════════════
# Convenience factory
# ═══════════════════════════════════════════════════════════════
def make_env(config) -> WorldCommEnv:
    return WorldCommEnv(config)