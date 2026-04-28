"""
Diagnose if the trained DQN adversary is exploiting a degenerate strategy
(e.g., pushing good agents into boundaries for negative reward instead of
actually catching them).

Usage:
    python diagnose_dqn.py
"""

import numpy as np
from config import Config
from envs import WorldCommEnv
from algorithms.dqn import DQNAgent
from frozen_policy import FrozenGoodPolicy


def diagnose(checkpoint_path, frozen_good_path, n_episodes=20):
    config = Config(
        algo="dqn",
        ablation_mode="full",
        num_good=2,
        num_forests=2,
        use_coop_reward=True,
        frozen_good_path=frozen_good_path,
    )

    env = WorldCommEnv(config)
    env.reset(seed=42)
    agent = DQNAgent(config, env)
    agent.load(checkpoint_path)

    good_specs = [env.get_agent_spec(a) for a in env.possible_agents
                  if a.startswith("agent_")]
    frozen = FrozenGoodPolicy(
        checkpoint_path=frozen_good_path,
        obs_dim=good_specs[0].obs_dim,
        action_dim=good_specs[0].action_dim,
        hidden=config.hidden_dim,
        device=config.device,
    )

    # Metrics
    adv_good_distances = []  # mean distance from adv to good each step
    collision_count = 0      # total collisions
    good_boundary_hits = 0   # times good agent at boundary (|x| > 0.9)
    adv_boundary_hits = 0
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 500)
        while env.agents:
            # Get actions
            adv_obs = {a: o for a, o in obs.items() if "adversary" in a}
            actions = agent.select_actions(adv_obs, env, explore=False)
            for g in [a for a in env.possible_agents if a.startswith("agent_")]:
                if g in obs and g in env.agents:
                    actions[g] = frozen.act(obs[g])
            actions = {a: actions[a] for a in env.agents if a in actions}

            obs, _, _, _, _ = env.step(actions)

            # Extract positions from world
            world = env._env.unwrapped.world
            pos = {a.name: np.asarray(a.state.p_pos) for a in world.agents}

            adv_names = [n for n in pos if "adversary" in n]
            good_names = [n for n in pos if "agent_" in n]

            # Average distance from each adv to nearest good
            if adv_names and good_names:
                dists = []
                for a in adv_names:
                    dists.append(min(np.linalg.norm(pos[a] - pos[g])
                                     for g in good_names))
                adv_good_distances.append(np.mean(dists))

            # Boundary hits
            for g in good_names:
                if np.any(np.abs(pos[g]) > 0.9):
                    good_boundary_hits += 1
            for a in adv_names:
                if np.any(np.abs(pos[a]) > 0.9):
                    adv_boundary_hits += 1

            # Collision count
            for a in adv_names:
                for g in good_names:
                    if np.linalg.norm(pos[a] - pos[g]) < 0.15:
                        collision_count += 1

            total_steps += 1

    env.close()

    # Report
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluated over {n_episodes} episodes ({total_steps} steps)")
    print("=" * 60)
    print(f"\nAverage adv->good distance:     {np.mean(adv_good_distances):.3f}")
    print(f"  (environment is 2x2; distance<0.5 = close chase)")
    print(f"\nCollisions (adv catches good): {collision_count}")
    print(f"  (avg per episode: {collision_count/n_episodes:.1f})")
    print(f"\nBoundary hits:")
    print(f"  good at boundary: {good_boundary_hits} steps "
          f"({100*good_boundary_hits/total_steps:.1f}%)")
    print(f"  adv at boundary:  {adv_boundary_hits} steps "
          f"({100*adv_boundary_hits/total_steps:.1f}%)")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if np.mean(adv_good_distances) > 0.6:
        print("[WARN] Adversaries are FAR from good agents on average.")
        print("       They are NOT actively pursuing.")
    else:
        print("[OK] Adversaries are close to good agents; they ARE pursuing.")

    if good_boundary_hits / total_steps > 0.4:
        print("[WARN] Good agents are at boundary > 40% of the time.")
        print("       Likely being pushed/herded by adversaries into boundary.")
        print("       This is the 'exploit frozen policy' failure mode.")
    else:
        print("[OK] Good agents mostly staying away from boundary.")

    if collision_count < n_episodes * 2:
        print("[WARN] Very few collisions. Adversary not catching good.")
    else:
        print("[OK] Reasonable number of collisions.")


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "outputs/models/test_new_dqn_full_coop_fg_s42/final"
    frozen = sys.argv[2] if len(sys.argv) > 2 else "outputs/models/standard_good_mappo_full_coop_s42/final"
    diagnose(ckpt, frozen, n_episodes=20)
