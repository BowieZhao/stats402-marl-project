"""
Entry point for MARL training on simple_world_comm_v3.

Usage:
    python main.py --algo mappo --total_episodes 6000 --seed 42
    python main.py --algo mappo --disable_comm   # comm ablation
"""

import argparse
import os
import traceback

from config import Config
from envs import WorldCommEnv
from experiment import ExperimentRunner


def parse_args():
    p = argparse.ArgumentParser(description="MARL training for simple_world_comm_v3")

    p.add_argument("--algo", type=str, default="mappo",
                   choices=["mappo", "dqn", "ddpg", "pso"],
                   help="Algorithm to run.")
    p.add_argument("--total_episodes", type=int, default=None,
                   help="Number of training episodes.")
    p.add_argument("--max_cycles", type=int, default=None,
                   help="Max environment steps per episode.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed.")
    p.add_argument("--run_name", type=str, default=None,
                   help="Run name used for logs/models.")
    p.add_argument("--render_mode", type=str, default=None,
                   help="Render mode for the environment.")
    p.add_argument("--ablation", type=str, default=None,
                   choices=["full", "no_comm", "no_leader_pos", "blind"],
                   help="Ablation mode: full/no_comm/no_leader_pos/blind")
    p.add_argument("--device", type=str, default=None,
                   help="Device: cpu or cuda.")

    return p.parse_args()


def build_agent(config, env):
    if config.algo == "mappo":
        from algorithms.mappo import MAPPOAgent
        return MAPPOAgent(config, env)
    if config.algo == "dqn":
        from algorithms.dqn import DQNAgent
        return DQNAgent(config, env)
    if config.algo == "ddpg":
        from algorithms.ddpg import DDPGAgent
        return DDPGAgent(config, env)
    if config.algo == "pso":
        from algorithms.pso import PSOAgent
        return PSOAgent(config, env)
    raise ValueError(f"Unsupported algo: {config.algo}")


def print_summary(config, env):
    print("=" * 50)
    print(f"  algo            = {config.algo}")
    print(f"  exp_name        = {config.exp_name}")
    print(f"  seed            = {config.seed}")
    print(f"  device          = {config.device}")
    print(f"  max_cycles      = {config.max_cycles}")
    print(f"  ablation_mode   = {config.ablation_mode}")
    print(f"  total_episodes  = {config.total_episodes}")
    print(f"  update_every    = {config.update_every_n_episodes}")
    print()
    print("  Agents:")
    for agent in env.possible_agents:
        spec = env.get_agent_spec(agent)
        print(f"    {agent:<20s}  role={spec.role:<10s}  "
              f"obs={spec.obs_dim}  act={spec.action_dim}  type={spec.action_type}")
    print("=" * 50)


def main():
    args = parse_args()

    # Build config — CLI args override defaults
    config = Config(algo=args.algo)
    if args.total_episodes is not None:
        config.total_episodes = args.total_episodes
    if args.max_cycles is not None:
        config.max_cycles = args.max_cycles
    if args.seed is not None:
        config.seed = args.seed
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.render_mode is not None:
        config.render_mode = args.render_mode
    if args.device is not None:
        config.device = args.device
    if args.ablation is not None:
        config.ablation_mode = args.ablation

    config.make_dirs()
    config.set_seed_everywhere()

    # Save config
    cfg_path = os.path.join(config.log_dir, f"{config.exp_name}_config.json")
    config.save_json(cfg_path)

    env = None
    try:
        env = WorldCommEnv(config)

        # One reset to initialise agent specs (required before build_agent)
        env.reset(seed=config.seed)

        print_summary(config, env)

        agent = build_agent(config, env)
        runner = ExperimentRunner(config, env, agent)
        runner.run()

        print(f"\nConfig saved to: {cfg_path}")

    except Exception as e:
        print("\n=== ERROR ===")
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
