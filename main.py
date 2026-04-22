"""
Entry point for MARL training on simple_world_comm_v3
"""

import argparse
import os
import traceback

from config import Config
from envs import WorldCommEnv
from experiment import ExperimentRunner


# =========================
# Args
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--algo", type=str, default="mappo",
                   choices=["mappo", "dqn", "ddpg", "ippo", "pso"])

    p.add_argument("--total_episodes", type=int, default=None)
    p.add_argument("--max_cycles", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--render_mode", type=str, default=None)
    p.add_argument("--ablation", type=str, default=None,
                   choices=["full", "no_comm", "no_leader_pos", "blind"])
    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


# =========================
# Agent Builder (统一接口！)
# =========================
def build_agent(config, env):

    algo = config.algo.lower()

    # -------------------------
    # MAPPO / IPPO
    # -------------------------
    if algo == "mappo":
        from algorithms.mappo import MAPPOAgent
        return MAPPOAgent(config, env)

    if algo == "ippo":
        from algorithms.ippo import IPPOAgent
        return IPPOAgent(config, env)

    # -------------------------
    # DQN (multi-agent)
    # -------------------------
    if algo == "dqn":
        from algorithms.dqn import DQNMultiAgent
        return DQNMultiAgent(env, config)

    # -------------------------
    # DDPG (multi-agent)
    # -------------------------
    if algo == "ddpg":
        from algorithms.ddpg import DDPGMultiAgent
        return DDPGMultiAgent(env, config)

    # -------------------------
    # PSO (baseline, non-RL)
    # -------------------------
    if algo == "pso":
        from algorithms.pso import PSOMultiAgent
        return PSOMultiAgent(env, config)

    raise ValueError(f"Unsupported algo: {algo}")


# =========================
# Print summary
# =========================
def print_summary(config, env):
    print("=" * 50)
    print(f"algo            = {config.algo}")
    print(f"exp_name        = {config.exp_name}")
    print(f"seed            = {config.seed}")
    print(f"device          = {config.device}")
    print(f"max_cycles      = {config.max_cycles}")
    print(f"ablation_mode   = {config.ablation_mode}")
    print(f"total_episodes  = {config.total_episodes}")
    print()

    print("Agents:")
    for agent in env.possible_agents:
        spec = env.get_agent_spec(agent)
        print(f"  {agent:<18} role={spec.role:<10} obs={spec.obs_dim} act={spec.action_dim}")
    print("=" * 50)


# =========================
# Main
# =========================
def main():
    args = parse_args()

    config = Config(algo=args.algo)

    # override
    if args.total_episodes:
        config.total_episodes = args.total_episodes
    if args.max_cycles:
        config.max_cycles = args.max_cycles
    if args.seed is not None:
        config.seed = args.seed
    if args.run_name:
        config.run_name = args.run_name
    if args.render_mode:
        config.render_mode = args.render_mode
    if args.device:
        config.device = args.device
    if args.ablation:
        config.ablation_mode = args.ablation

    config.make_dirs()
    config.set_seed_everywhere()

    # save config
    cfg_path = os.path.join(config.log_dir, f"{config.exp_name}_config.json")
    config.save_json(cfg_path)

    env = None

    try:
        env = WorldCommEnv(config)
        env.reset(seed=config.seed)

        print_summary(config, env)

        agent = build_agent(config, env)
        runner = ExperimentRunner(config, env, agent)

        runner.run()

        print(f"\nSaved config: {cfg_path}")

    except Exception as e:
        print("\n=== ERROR ===")
        traceback.print_exc()
        raise

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()