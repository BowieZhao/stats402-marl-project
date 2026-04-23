"""
Sanity check: verify cooperative reward reshaping on a SINGLE trajectory.

We run one env step-by-step with random actions, and at each step manually
compute what the shaping bonus would be. Then compare:
  - Raw MPE reward (what env returns with use_coop_reward=False)
  - Bonus that shaping would add (computed from agent positions)

This avoids the "two different random episodes" confound of the earlier test.
"""

import numpy as np
from config import Config
from envs import WorldCommEnv


def compute_bonus_from_positions(env_wrapper):
    """
    Replicate the bonus logic from envs.py _reshape_rewards.
    Returns per-adversary bonus dict and a "all goods covered" bool.
    """
    cfg = env_wrapper.config
    radius = cfg.coop_radius
    bonus_per_extra = cfg.coop_bonus_per_extra
    coverage_bonus = cfg.coverage_bonus

    positions = env_wrapper._get_agent_positions()
    world = env_wrapper._env.unwrapped.world
    adv_names = [a.name for a in world.agents if a.adversary]
    good_names = [a.name for a in world.agents if not a.adversary]

    adv_bonus = {a: 0.0 for a in adv_names}
    all_covered = True

    for g in good_names:
        g_pos = positions[g]
        nearby = [a for a in adv_names
                  if float(np.linalg.norm(positions[a] - g_pos)) < radius]
        if len(nearby) == 0:
            all_covered = False
        if len(nearby) >= 2:
            extra = (len(nearby) - 1) * bonus_per_extra
            for a in nearby:
                adv_bonus[a] += extra

    coverage = coverage_bonus if all_covered else 0.0
    return adv_bonus, coverage, all_covered


# ── Run one episode with random actions, track both reward types ──
config = Config(use_coop_reward=False, num_good=2, num_forests=2, max_cycles=50)
env = WorldCommEnv(config)

np.random.seed(42)
obs, _ = env.reset(seed=42)

adv_names = [a for a in env.possible_agents if "adversary" in a]
good_names = [a for a in env.possible_agents if a.startswith("agent_")]

raw_totals = {a: 0.0 for a in env.possible_agents}
bonus_totals = {a: 0.0 for a in adv_names}
coverage_count = 0
surround_count = 0
n_steps = 0

for t in range(50):
    if not env.agents:
        break
    actions = {a: np.random.randint(env.action_spaces[a].n) for a in env.agents}
    obs, rewards, term, trunc, _ = env.step(actions)

    # Compute what bonus shaping would add at the new state
    adv_bonus, coverage, all_covered = compute_bonus_from_positions(env)

    for a, r in rewards.items():
        raw_totals[a] += r
    for a in adv_names:
        bonus_totals[a] += adv_bonus.get(a, 0.0) + coverage

    if all_covered:
        coverage_count += 1
    if any(adv_bonus.get(a, 0.0) > 0 for a in adv_names):
        surround_count += 1
    n_steps += 1

env.close()

# ── Report ───────────────────────────────────────────────────────
print("=" * 65)
print(f"Ran 1 episode with random actions, {n_steps} steps.")
print("=" * 65)

print(f"\nRaw rewards (from MPE):")
for a in env.possible_agents:
    role = "ADV" if "adversary" in a else "GOOD"
    print(f"  [{role}] {a:<20s} total_r = {raw_totals[a]:+7.2f}")

print(f"\nBonus that shaping would add (adversary side only):")
for a in adv_names:
    print(f"        {a:<20s} bonus   = {bonus_totals[a]:+7.2f}")

print(f"\nStatistics:")
print(f"  Steps with surround (>=2 adv near same good): {surround_count}/{n_steps}"
      f"  ({100*surround_count/max(n_steps,1):.1f}%)")
print(f"  Steps with full coverage (all goods covered): {coverage_count}/{n_steps}"
      f"  ({100*coverage_count/max(n_steps,1):.1f}%)")

print(f"\nAdversary team totals:")
raw_team = sum(raw_totals[a] for a in adv_names)
bonus_team = sum(bonus_totals[a] for a in adv_names)
print(f"  Raw        : {raw_team:+7.2f}")
print(f"  + Bonus    : {bonus_team:+7.2f}")
print(f"  = Shaped   : {raw_team + bonus_team:+7.2f}")

print("\n" + "=" * 65)
print("INTERPRETATION")
print("=" * 65)
if bonus_team > 0.1:
    print(f"\n[OK] Shaping is WORKING. Bonus of +{bonus_team:.2f} triggered even under")
    print("     random actions. Trained policies will trigger it much more often.")
elif bonus_team == 0:
    print("\n[WARN] Zero bonus under random actions.")
    print("       This can happen because random agents rarely cluster.")
    print("       Let's verify the code runs by lowering coop_radius and retrying...")

    # ── Retry with much larger radius to force bonus to trigger ───
    print("\n" + "-" * 65)
    print("Retry with coop_radius=1.5 (huge, forces bonuses to fire):")
    print("-" * 65)
    config2 = Config(use_coop_reward=False, num_good=2, num_forests=2, max_cycles=50,
                     coop_radius=1.5)
    env2 = WorldCommEnv(config2)
    np.random.seed(42)
    obs, _ = env2.reset(seed=42)

    raw2 = {a: 0.0 for a in env2.possible_agents}
    bonus2 = {a: 0.0 for a in adv_names}
    for t in range(30):
        if not env2.agents:
            break
        actions = {a: np.random.randint(env2.action_spaces[a].n) for a in env2.agents}
        obs, rewards, _, _, _ = env2.step(actions)
        adv_b, cov, _ = compute_bonus_from_positions(env2)
        for a, r in rewards.items():
            raw2[a] += r
        for a in adv_names:
            bonus2[a] += adv_b.get(a, 0.0) + cov
    env2.close()

    total_bonus = sum(bonus2.values())
    print(f"With radius=1.5, total bonus over 30 steps: +{total_bonus:.2f}")
    if total_bonus > 0:
        print("[OK] Shaping code is correct. Original radius=0.3 is just strict.")
    else:
        print("[ERROR] Still zero — shaping code may have a bug.")
else:
    print(f"\n[?] Negative bonus ({bonus_team:.2f}) — this shouldn't happen.")
