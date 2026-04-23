"""
Extended diagnosis: find which obs dims correspond to good agents.

Run: python diagnose2.py
"""

import numpy as np
from config import Config
from envs import WorldCommEnv, is_leader, is_normal_adversary, is_good


def find_good_agent_dims():
    """
    Strategy: reset twice with different seeds, same adversary position
    shouldn't matter — what matters is that good agent positions are different.
    We identify which dims in adversary obs change based on good agent position.
    """
    print("=" * 60)
    print("IDENTIFYING GOOD AGENT DIMS IN ADVERSARY OBS")
    print("=" * 60)

    config = Config()
    env = WorldCommEnv(config)

    # Strategy: compare obs across many seeds, track which dims have
    # consistent structure. Good agents contribute to "other_agent_rel_pos"
    # which is in dims [14:24] (5 agents × 2 coords each).
    # But WHICH of those 5 are good vs adversary?

    # There are 5 "other agents" from each adversary's POV:
    # - 1 leader
    # - 2 other normal adversaries
    # - 2 good agents
    # So dims [14:24] = 10 dims for 5 agents × 2 coords

    # MPE orders agents: adversaries first, then good agents
    # So for adversary_0's view of "other agents":
    #   [leader, adv_1, adv_2, good_0, good_1]
    # Positions: [14:16]=leader, [16:18]=adv_1, [18:20]=adv_2,
    #            [20:22]=good_0, [22:24]=good_1

    obs, _ = env.reset(seed=42)

    # Print adversary_0's obs with per-dim breakdown
    adv = "adversary_0"
    ob = obs[adv]

    print(f"\n{adv} obs ({len(ob)} dims) at reset(seed=42):")
    print(f"  [0:2]    self_vel       = {ob[0:2]}")
    print(f"  [2:4]    self_pos       = {ob[2:4]}")
    print(f"  [4:6]    landmark_0     = {ob[4:6]}  (obstacle)")
    print(f"  [6:8]    landmark_1     = {ob[6:8]}  (food?)")
    print(f"  [8:10]   landmark_2     = {ob[8:10]} (food?)")
    print(f"  [10:12]  landmark_3     = {ob[10:12]} (forest?)")
    print(f"  [12:14]  landmark_4     = {ob[12:14]} (forest?)")
    print(f"  [14:16]  other_agent_0  = {ob[14:16]} (likely LEADER)")
    print(f"  [16:18]  other_agent_1  = {ob[16:18]} (likely adversary)")
    print(f"  [18:20]  other_agent_2  = {ob[18:20]} (likely adversary)")
    print(f"  [20:22]  other_agent_3  = {ob[20:22]} (likely GOOD_0)")
    print(f"  [22:24]  other_agent_4  = {ob[22:24]} (likely GOOD_1)")
    print(f"  [24:26]  other_vel_0    = {ob[24:26]}")
    print(f"  [26:28]  other_vel_1    = {ob[26:28]}")
    print(f"  [28:30]  self_in_forest = {ob[28:30]}")
    print(f"  [30:34]  leader_comm    = {ob[30:34]}")

    # Verify by checking the good agent's position from its own obs
    # and comparing with what adv sees
    good = "agent_0"
    if good in obs:
        g_ob = obs[good]
        # Good agent obs: [self_vel(2), self_pos(2), landmark_pos(10),
        #                  other_agent_pos(8), self_in_forest(2), other_vel(4?)]
        # Good's own position is at [2:4]
        good_pos = g_ob[2:4]
        adv_pos = ob[2:4]
        # Relative position of good from adv's perspective = good_pos - adv_pos
        expected_rel = good_pos - adv_pos
        print(f"\n  Sanity check:")
        print(f"  {good} absolute pos  = {good_pos}")
        print(f"  {adv} absolute pos   = {adv_pos}")
        print(f"  Expected rel_pos      = {expected_rel}")
        print(f"  Looking for this in adv obs...")
        # Search for this value in the other_agent section
        for start in range(14, 24, 2):
            candidate = ob[start:start+2]
            if np.allclose(candidate, expected_rel, atol=1e-4):
                print(f"  ✓ MATCH found at obs[{start}:{start+2}] — this IS good_0")
                break
        else:
            print(f"  (not found at exact match; may be sorted differently)")

    env.close()


def verify_good_visibility():
    """
    Run many episodes, check how often good agents are actually
    occluded from adversary observations (i.e., relative position = (0,0)).
    """
    print("\n" + "=" * 60)
    print("GOOD AGENT OCCLUSION FREQUENCY")
    print("=" * 60)

    config = Config(max_cycles=50, num_forests=2)
    env = WorldCommEnv(config)

    # Assuming dims [20:22] and [22:24] are the two good agents for normal adv
    good_occlusion = {"good_0": 0, "good_1": 0, "both": 0, "any": 0}
    total_steps = 0

    for ep in range(30):
        obs, _ = env.reset(seed=ep)
        while env.agents:
            actions = {}
            for a in env.agents:
                act_space = env.action_spaces[a]
                if hasattr(act_space, "n"):
                    actions[a] = np.random.randint(act_space.n)
                else:
                    actions[a] = act_space.sample()
            obs, _, _, _, _ = env.step(actions)

            # Check adversary_0's view
            if "adversary_0" in obs:
                ob = obs["adversary_0"]
                g0 = ob[20:22]
                g1 = ob[22:24]

                g0_occluded = np.allclose(g0, 0, atol=1e-6)
                g1_occluded = np.allclose(g1, 0, atol=1e-6)

                if g0_occluded:
                    good_occlusion["good_0"] += 1
                if g1_occluded:
                    good_occlusion["good_1"] += 1
                if g0_occluded and g1_occluded:
                    good_occlusion["both"] += 1
                if g0_occluded or g1_occluded:
                    good_occlusion["any"] += 1
                total_steps += 1

    print(f"\nOver {total_steps} steps (30 episodes):")
    for k, v in good_occlusion.items():
        pct = 100 * v / total_steps if total_steps > 0 else 0
        print(f"  {k:10s}: {v}/{total_steps} ({pct:.1f}%)")

    print("\nInterpretation:")
    print("  If 'any' is < 20%, forest occlusion is too weak:")
    print("  adversary can see good agents directly most of the time,")
    print("  which explains why ablations don't matter.")
    print("\nRecommendation: increase num_forests to 4-6.")

    env.close()


def test_more_forests():
    """Show that more forests → more occlusion."""
    print("\n" + "=" * 60)
    print("EFFECT OF INCREASING NUM_FORESTS")
    print("=" * 60)

    for nf in [2, 4, 6, 8]:
        config = Config(max_cycles=50, num_forests=nf)
        # Need to re-init env since num_forests is passed in
        try:
            env = WorldCommEnv(config)
            occluded = 0
            total = 0
            for ep in range(10):
                obs, _ = env.reset(seed=ep)
                while env.agents:
                    actions = {a: np.random.randint(env.action_spaces[a].n)
                               for a in env.agents}
                    obs, _, _, _, _ = env.step(actions)
                    if "adversary_0" in obs:
                        ob = obs["adversary_0"]
                        g0 = ob[20:22]
                        g1 = ob[22:24]
                        if np.allclose(g0, 0, atol=1e-6) or np.allclose(g1, 0, atol=1e-6):
                            occluded += 1
                        total += 1
            pct = 100 * occluded / total if total else 0
            print(f"  num_forests={nf}: {occluded}/{total} steps have occlusion ({pct:.1f}%)")
            env.close()
        except Exception as e:
            print(f"  num_forests={nf}: error {e}")


if __name__ == "__main__":
    find_good_agent_dims()
    verify_good_visibility()
    test_more_forests()
