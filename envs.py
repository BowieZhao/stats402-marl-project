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
        """Set of good names that are currently colliding with at least one adv."""
        positions = self._get_agent_positions()
        adv_names = self._leader_names + self._normal_adv_names
        good_names = self._good_names

        caught = set()
        for g in good_names:
            if g not in positions:
                continue
            for a in adv_names:
                if a not in positions:
                    continue
                d = float(np.linalg.norm(positions[a] - positions[g]))
                if d < self.config.collision_distance:
                    caught.add(g)
                    break  # one adv catching this good is enough
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
        Compute the structured reward stack.
        Returns dict of {agent_name: reward} for adversaries (leader + normal).

        Reward components (per-step, per-adv):

          R_distance       -0.05 × dist(adv, nearest UNCAUGHT good)
                            Goes to 0 if all goods are caught.

          R_capture        +10 if this adv collided with an UNCAUGHT good
                            (already-caught goods give nothing — forces
                            switching to next target)

          R_encircle       +(n-1)*4 if adv is in encircle group of an UNCAUGHT
                            good with n ≥ 3 advs within 0.4 radius

          R_progress       +30 if adv is the killer of the 1st-caught good
                           +40 if adv is the killer of the 2nd-caught good
                            (second-catch escalation reflects time pressure)
                            Only the actual collider gets this — no team-share.

          R_pin_assist     +10 if a good was first-caught this step AND
                            this adv is in the encircle group (within 0.4)
                            BUT is not the killer.
                            Rewards meaningful encircle participation.

          R_team_bonus     +5 to everyone if any good was first-caught this step
                            Light celebration, doesn't dilute other signals.

          R_role_align     +1 if the action matches plan_bias top direction

        Leader extra:
          R_thrash         -3 if message switched without justification
        """
        cfg = self.config
        positions = self._get_agent_positions()
        adv_names = self._leader_names + self._normal_adv_names
        good_names = self._good_names

        if not adv_names or not good_names:
            return {a: 0.0 for a in adv_names}

        rewards = {a: 0.0 for a in adv_names}

        # Determine which goods are still uncaught (haven't been first-caught yet)
        uncaught_goods = [g for g in good_names if g not in self._goods_caught_ever]

        # ── (1) R_distance: only toward UNCAUGHT goods ──
        for a in adv_names:
            if a not in positions:
                continue
            if not uncaught_goods:
                continue  # all caught, no distance reward
            min_d = min(np.linalg.norm(positions[a] - positions[g])
                        for g in uncaught_goods if g in positions)
            rewards[a] += cfg.R_distance_coef * float(min_d)

        # ── (2) R_capture: only on UNCAUGHT good collisions ──
        # NB: at this point, _goods_caught_ever already includes new_first_catches
        # (set in step()), so "uncaught" excludes the goods just first-caught
        # this step. To award R_capture to the killer, we need the snapshot
        # BEFORE this step's first-catches were added. We use _caught_set_this_step
        # together with the previous goods_caught_ever (= goods_caught_ever - new).
        prev_goods_caught_ever = self._goods_caught_ever - new_first_catches
        for a in adv_names:
            if a not in positions:
                continue
            for g in good_names:
                if g not in positions:
                    continue
                if g in prev_goods_caught_ever:
                    continue  # already caught before this step → no capture reward
                d = float(np.linalg.norm(positions[a] - positions[g]))
                if d < cfg.collision_distance:
                    rewards[a] += cfg.R_capture
                    break  # one collision per adv per step

        # ── (3) R_encircle: only on UNCAUGHT goods ──
        for g in good_names:
            if g not in positions:
                continue
            if g in prev_goods_caught_ever:
                continue   # don't reward encircling already-caught goods
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

        # ── (4) R_progress + R_pin_assist + R_team_bonus on first catches ──
        for caught_g in new_first_catches:
            if caught_g not in positions:
                continue
            g_pos = positions[caught_g]

            # Identify the killer: adv that's actually colliding
            killer = None
            for a in adv_names:
                if a not in positions:
                    continue
                d = float(np.linalg.norm(positions[a] - g_pos))
                if d < cfg.collision_distance:
                    killer = a
                    break

            # Identify rank of this catch (1st or 2nd)
            # Number of catches BEFORE this one = (already-caught goods - this catch)
            n_already = len(self._goods_caught_ever) - 1   # this catch is in the set
            # n_already=0 means this is the 1st catch; n_already=1 → 2nd catch
            if n_already == 0:
                progress_reward = cfg.R_progress_first
            else:
                progress_reward = cfg.R_progress_second

            # Award R_progress to killer
            if killer is not None and killer in rewards:
                rewards[killer] += progress_reward

            # Award R_pin_assist + R_progress_team_share to encircle group members
            # (split-incentive: killer gets the big share, encircle teammates get a
            #  meaningful fraction so total team reward stays high enough to learn)
            for a in adv_names:
                if a not in positions or a == killer:
                    continue
                d = float(np.linalg.norm(positions[a] - g_pos))
                if d < cfg.R_encircle_radius:
                    rewards[a] += cfg.R_pin_assist
                    rewards[a] += cfg.R_progress_team_share

            # Award R_team_bonus to everyone
            for a in adv_names:
                rewards[a] += cfg.R_team_bonus

        # ── (5) R_role_align ── (TEACHING reward — skipped in eval mode)
        if cfg.plan_bias_enabled and not self._eval_mode:
            for a in self._normal_adv_names:
                if a not in action_dict:
                    continue
                action = int(action_dict[a])
                if a in positions:
                    base_obs = self._reconstruct_adv_base_obs(a, positions)
                    bias = self.compute_plan_bias(leader_message, base_obs)
                    if np.max(bias) > 0 and action == int(np.argmax(bias)):
                        rewards[a] += cfg.R_role_align

        # ── (6) R_thrash ── (TEACHING reward — skipped in eval mode)
        if cfg.message_enabled and leader_message != self._last_message and not self._eval_mode:
            current_forest_state = self._compute_forest_state()
            justified = (
                len(self._caught_set_this_step) > 0 or
                current_forest_state != self._last_forest_state
            )
            if not justified:
                for ln in self._leader_names:
                    if ln in rewards:
                        rewards[ln] += cfg.R_message_thrash_penalty

        # ── (7) R_correct_message ── (TEACHING reward — skipped in eval mode)
        if cfg.message_enabled and cfg.use_correct_message_reward and not self._eval_mode:
            if leader_message != self._last_message:
                correct_msg = self._compute_correct_message()
                forest_state = self._compute_forest_state()
                any_in_forest = any(forest_state)
                uncaught = [g for g in self._good_names
                            if g not in self._goods_caught_ever]

                # Determine reward only for "meaningful" correct switches:
                # - Forest case: switching INTO C/D when good entered forest
                # - One-good-left case: switching to focus the remaining good
                # - Both-exposed case: NOT rewarded (A↔B is equivalent)
                is_meaningful_correct = False
                if any_in_forest:
                    # Reward only if new message is the correct C/D
                    is_meaningful_correct = (leader_message == correct_msg)
                elif len(uncaught) == 1:
                    # Reward only if new message focuses on remaining good
                    is_meaningful_correct = (leader_message == correct_msg)
                # else (both exposed): no reward — A and B are equivalent

                if is_meaningful_correct:
                    for ln in self._leader_names:
                        if ln in rewards:
                            rewards[ln] += cfg.R_correct_message

        return rewards

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
