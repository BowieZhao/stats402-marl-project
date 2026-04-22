"""
Diagnostic script: inspect what's actually happening in the environment.

Run: python diagnose.py

This checks:
1. What dimensions of normal adversary obs correspond to leader_comm
2. Whether forests actually occlude good agents in practice
3. Whether the leader's comm messages are differentiated
4. What information normal adversaries get even without comm
"""

import numpy as np
from config import Config
from envs import WorldCommEnv, is_leader, is_normal_adversary, is_good


def inspect_obs_structure():
    """
    Compare obs between two resets to figure out which dims change
    when leader sends different messages.
    """
    print("=" * 60)
    print("1. OBSERVATION STRUCTURE INSPECTION")
    print("=" * 60)

    config = Config()
    env = WorldCommEnv(config)
    obs, _ = env.reset(seed=42)

    for agent in env.possible_agents:
        spec = env.get_agent_spec(agent)
        print(f"\n{agent} (role={spec.role}, obs_dim={spec.obs_dim}, act_dim={spec.action_dim})")
        print(f"  obs = {obs[agent]}")
        print(f"  obs range: [{obs[agent].min():.3f}, {obs[agent].max():.3f}]")

    env.close()


def inspect_comm_dims():
    """
    Run two episodes with different leader comm actions,
    check which obs dims change for normal adversaries.
    """
    print("\n" + "=" * 60)
    print("2. IDENTIFYING COMMUNICATION DIMENSIONS")
    print("=" * 60)

    config = Config()
    env = WorldCommEnv(config)

    # Run with leader sending message 0 (action 0-4 = say_0 × moves)
    obs1, _ = env.reset(seed=42)

    # All agents take action 0 except leader takes action 0 (say_0, no_move)
    actions_say0 = {}
    for a in env.agents:
        if is_leader(a):
            actions_say0[a] = 0  # say_0 × no_action
        else:
            actions_say0[a] = 0  # no_action

    next_obs1, _, _, _, _ = env.step(actions_say0)

    # Reset and do same but leader sends message 3 (action 15-19 = say_3 × moves)
    obs2, _ = env.reset(seed=42)

    actions_say3 = {}
    for a in env.agents:
        if is_leader(a):
            actions_say3[a] = 15  # say_3 × no_action
        else:
            actions_say3[a] = 0

    next_obs2, _, _, _, _ = env.step(actions_say3)

    # Compare normal adversary obs
    print("\nComparing normal adversary obs after leader says 0 vs says 3:")
    for a in env.possible_agents:
        if is_normal_adversary(a) and a in next_obs1 and a in next_obs2:
            diff = next_obs1[a] - next_obs2[a]
            changed_dims = np.where(np.abs(diff) > 1e-6)[0]
            print(f"\n  {a}:")
            print(f"    Dims that changed: {changed_dims}")
            print(f"    obs (say_0): last 8 dims = {next_obs1[a][-8:]}")
            print(f"    obs (say_3): last 8 dims = {next_obs2[a][-8:]}")
            if len(changed_dims) > 0:
                print(f"    Diff at changed dims: {diff[changed_dims]}")
                print(f"    ==> COMM DIMS ARE: {changed_dims}")
                print(f"    ==> COMM DIM COUNT: {len(changed_dims)}")
            else:
                print(f"    WARNING: No dims changed! Comm may not be in obs.")

    env.close()


def inspect_forest_occlusion():
    """
    Run an episode and check how often good agents are occluded.
    When occluded, their relative position in adversary obs should be (0,0).
    """
    print("\n" + "=" * 60)
    print("3. FOREST OCCLUSION FREQUENCY")
    print("=" * 60)

    config = Config(max_cycles=50)
    env = WorldCommEnv(config)

    total_steps = 0
    occlusion_counts = {a: 0 for a in env.possible_agents if is_normal_adversary(a)}

    for episode in range(20):
        obs, _ = env.reset(seed=episode)
        step = 0
        while env.agents:
            # Random actions
            actions = {}
            for a in env.agents:
                act_space = env.action_spaces[a]
                if hasattr(act_space, 'n'):
                    actions[a] = np.random.randint(act_space.n)
                else:
                    actions[a] = act_space.sample()

            next_obs, _, _, _, _ = env.step(actions)

            # Check if any good agent position appears as (0,0) in adversary obs
            # Good agent relative positions are somewhere in the obs vector
            # We look for consecutive (0,0) pairs that might indicate occlusion
            for adv in env.agents:
                if is_normal_adversary(adv) and adv in next_obs:
                    ob = next_obs[adv]
                    # Check for zero pairs in the "other agent relative positions" region
                    # These start after self_vel(2) + self_pos(2) + landmark_positions(?)
                    # Look for any (0,0) pair in dims 4-30 (rough estimate)
                    for i in range(4, len(ob) - 1, 2):
                        if abs(ob[i]) < 1e-6 and abs(ob[i+1]) < 1e-6:
                            occlusion_counts[adv] += 1
                            break  # count once per step per adversary

            obs = next_obs
            step += 1
            total_steps += 1

    print(f"\nOver {total_steps} total steps across 20 episodes:")
    for adv, count in occlusion_counts.items():
        pct = 100.0 * count / total_steps if total_steps > 0 else 0
        print(f"  {adv}: {count} steps with at least one (0,0) pair ({pct:.1f}%)")
    print("\n  Note: (0,0) pairs could also appear for non-occlusion reasons")
    print("  (e.g., agents at same position, or landmark at origin)")

    env.close()


def inspect_leader_visibility():
    """
    Key question: can normal adversaries see the leader's position and velocity?
    If yes, they can follow the leader even without comm — the leader's
    MOVEMENT is itself an implicit communication channel.
    """
    print("\n" + "=" * 60)
    print("4. IMPLICIT COMMUNICATION CHECK")
    print("=" * 60)
    print("\nNormal adversary obs structure (from MPE2 docs):")
    print("  [self_vel, self_pos, landmark_rel_positions,")
    print("   other_agent_rel_positions, other_agent_velocities,")
    print("   self_in_forest, leader_comm]")
    print()
    print("CRITICAL QUESTION: 'other_agent_rel_positions' includes the LEADER.")
    print("This means normal adversaries can always see WHERE the leader is,")
    print("and WHERE the leader is moving (via other_agent_velocities).")
    print()
    print("If the leader chases good agents, normal adversaries can just")
    print("FOLLOW THE LEADER — no explicit comm needed.")
    print("The leader's trajectory IS the communication.")
    print()
    print("This may explain why comm ON ≈ comm OFF in performance.")

    config = Config()
    env = WorldCommEnv(config)
    obs, _ = env.reset(seed=42)

    # Print breakdown attempt
    for a in env.possible_agents:
        if is_normal_adversary(a):
            ob = obs[a]
            print(f"\n{a} obs ({len(ob)} dims):")
            print(f"  [0:2]  self_vel       = {ob[0:2]}")
            print(f"  [2:4]  self_pos       = {ob[2:4]}")
            # The rest depends on entity counts
            # 5 landmarks (1 obstacle + 2 food + 2 forests) × 2 = 10
            print(f"  [4:14] landmark_rel   = {ob[4:14]}")
            # 5 other agents × 2 = 10
            print(f"  [14:24] agent_rel_pos = {ob[14:24]}")
            # 5 other agents × 2 = 10
            print(f"  [24:34] agent_vel     = {ob[24:34]}")
            # self_in_forest and leader_comm would need more dims...
            # but obs is only 34, so the breakdown must be different
            print(f"  Total accounted: 34 dims")
            print(f"  NOTE: self_in_forest and leader_comm must be")
            print(f"  squeezed into these 34 dims somehow.")
            break

    env.close()


def inspect_trained_leader_comm():
    """
    Load a trained checkpoint and see what messages the leader sends.
    """
    print("\n" + "=" * 60)
    print("5. TRAINED LEADER COMMUNICATION PATTERN")
    print("=" * 60)

    import os
    ckpt_path = "outputs/models/swc_mappo_mappo_comm_s42/final"
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found at {ckpt_path}, skipping.")
        return

    config = Config()
    env = WorldCommEnv(config)
    env.reset(seed=42)

    from algorithms.mappo import MAPPOAgent
    agent = MAPPOAgent(config, env)
    agent.load(ckpt_path)

    # Leader's 20 actions = 4 messages × 5 moves
    # action // 5 = message index, action % 5 = move index
    message_counts = [0, 0, 0, 0]
    move_counts = [0, 0, 0, 0, 0]
    total = 0

    for ep in range(50):
        obs, _ = env.reset(seed=ep + 100)
        while env.agents:
            actions = agent.select_actions(obs, env, explore=False)
            for a, act in actions.items():
                if is_leader(a):
                    msg = act // 5
                    mov = act % 5
                    message_counts[msg] += 1
                    move_counts[mov] += 1
                    total += 1

            active = {a: actions[a] for a in env.agents if a in actions}
            obs, _, _, _, _ = env.step(active)

    print(f"\nLeader action distribution over {total} steps (50 eval episodes):")
    print(f"  Messages: {[f'say_{i}: {c} ({100*c/total:.1f}%)' for i, c in enumerate(message_counts)]}")
    print(f"  Moves:    {[f'move_{i}: {c} ({100*c/total:.1f}%)' for i, c in enumerate(move_counts)]}")

    # Check if messages are uniform (= not learned meaningful comm)
    msg_probs = np.array(message_counts) / total
    msg_entropy = -np.sum(msg_probs * np.log(msg_probs + 1e-10))
    max_entropy = np.log(4)
    print(f"\n  Message entropy: {msg_entropy:.3f} (max={max_entropy:.3f})")
    if msg_entropy > 0.95 * max_entropy:
        print("  ==> Messages are nearly UNIFORM — leader has NOT learned")
        print("      to use communication meaningfully.")
    else:
        print(f"  ==> Some message differentiation detected.")

    env.close()


if __name__ == "__main__":
    inspect_obs_structure()
    inspect_comm_dims()
    inspect_forest_occlusion()
    inspect_leader_visibility()
    inspect_trained_leader_comm()

    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    print("""
If comm ON ≈ comm OFF, the most likely reasons are:

1. WRONG COMM DIMS: We assumed the last 4 dims are leader_comm,
   but they might be elsewhere or have different size.
   → Check section 2 output above.

2. IMPLICIT COMMUNICATION: Normal adversaries can see the leader's
   position and velocity. If the leader moves toward good agents,
   adversaries can follow the leader without needing explicit messages.
   → This is a fundamental env design issue, not a code bug.

3. FORESTS DON'T OCCLUDE ENOUGH: Good agents may rarely be in forests,
   so partial observability isn't strong enough to make comm necessary.
   → Check section 3 output above.

4. LEADER HASN'T LEARNED MEANINGFUL COMM: If message distribution is
   uniform, the leader is sending random noise regardless of state.
   → Check section 5 output above.
""")