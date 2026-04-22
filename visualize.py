"""
Load a trained checkpoint and replay episodes in render_mode="human".

Usage:
    python visualize.py --checkpoint outputs/models/swc_mappo_mappo_comm_s42/final
    python visualize.py --checkpoint outputs/models/swc_mappo_mappo_comm_s42/final --episodes 5
    python visualize.py --checkpoint outputs/models/swc_mappo_mappo_comm_s42/final --disable_comm
"""

import argparse
import time
import numpy as np
from collections import defaultdict

from config import Config
from envs import WorldCommEnv


def parse_args():
    p = argparse.ArgumentParser(description="Visualise trained agents")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint directory (containing leader.pt, adversary.pt, etc.)")
    p.add_argument("--episodes", type=int, default=3,
                   help="Number of episodes to replay.")
    p.add_argument("--max_cycles", type=int, default=50)
    p.add_argument("--ablation", type=str, default="full",
                   choices=["full", "no_comm", "no_leader_pos", "blind"],
                   help="Ablation mode for visualisation.")
    p.add_argument("--slow", type=float, default=0.04,
                   help="Seconds to sleep between steps (controls replay speed).")
    return p.parse_args()


def main():
    args = parse_args()

    config = Config(
        render_mode="human",
        max_cycles=args.max_cycles,
        ablation_mode=args.ablation,
    )

    env = WorldCommEnv(config)
    env.reset(seed=config.seed)

    # Build agent and load checkpoint
    from algorithms.mappo import MAPPOAgent
    agent = MAPPOAgent(config, env)
    agent.load(args.checkpoint)
    print(f"[Vis] Loaded checkpoint from {args.checkpoint}")
    print(f"[Vis] ablation_mode = {config.ablation_mode}")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_rewards = defaultdict(float)
        step = 0

        while env.agents:
            actions = agent.select_actions(obs, env, explore=False)
            active_actions = {a: actions[a] for a in env.agents if a in actions}
            next_obs, rewards, terminations, truncations, infos = env.step(active_actions)

            for a, r in rewards.items():
                ep_rewards[a] += r

            obs = next_obs
            step += 1

            if args.slow > 0:
                time.sleep(args.slow)

        # Print episode summary
        adv_r = np.mean([ep_rewards[a] for a in agent.adv_agents if a in ep_rewards])
        good_r = np.mean([ep_rewards[a] for a in agent.good_agents if a in ep_rewards])
        print(f"  [Episode {ep}]  steps={step}  adv_reward={adv_r:.2f}  good_reward={good_r:.2f}")

    env.close()
    print("[Vis] Done.")


if __name__ == "__main__":
    main()
