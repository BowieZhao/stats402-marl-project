"""
Main entry point for Stage 3 experiments.

Supports algorithms:
  --algo mappo : MAPPO with ABCD messages, plan_bias, alpha-gated actor
  --algo dqn   : Independent DQN baseline (role-shared params)
  --algo ddpg  : DDPG with Gumbel-softmax discrete actions
  --algo pso   : PSO over policy parameters

Conditions (apply to all algorithms):
  E1_full     : message + plan_bias + (learnable alpha for mappo)
  E2_no_comm  : no message, no plan_bias
  E3_no_alpha : message + plan_bias, alpha fixed at 0.5
"""

import argparse
import os
import sys
from config import Config
from envs import make_env
from experiment import Runner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="mappo",
                   choices=["mappo", "dqn", "ddpg", "pso"],
                   help="Algorithm to use")
    p.add_argument("--condition", default="E1_full",
                   choices=["E1_full", "E2_no_comm", "E3_no_alpha"],
                   help="Experiment condition")
    p.add_argument("--total_episodes", type=int, default=4000)
    p.add_argument("--max_cycles", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default="exp")
    p.add_argument("--num_good", type=int, default=2)
    p.add_argument("--num_forests", type=int, default=2)
    p.add_argument("--frozen_good", default=None)
    p.add_argument("--frozen_good_deterministic", action="store_true")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def build_agent(cfg, env):
    """Dispatch to the right agent class based on cfg.algo."""
    if cfg.algo == "mappo":
        from algorithms.mappo import MAPPOAgent
        return MAPPOAgent(cfg, env)
    elif cfg.algo == "dqn":
        from algorithms.dqn import DQNAgent
        return DQNAgent(cfg, env)
    elif cfg.algo == "ddpg":
        from algorithms.ddpg import DDPGAgent
        return DDPGAgent(cfg, env)
    elif cfg.algo == "pso":
        from algorithms.pso import PSOAgent
        return PSOAgent(cfg, env)
    else:
        raise ValueError(f"Unknown algo: {cfg.algo}")


def main():
    args = parse_args()

    cfg = Config()
    cfg.algo = args.algo
    cfg.condition = args.condition
    cfg.total_episodes = args.total_episodes
    cfg.max_cycles = args.max_cycles
    cfg.seed = args.seed
    cfg.run_name = args.run_name
    cfg.num_good = args.num_good
    cfg.num_forests = args.num_forests
    cfg.frozen_good_path = args.frozen_good
    cfg.frozen_good_deterministic = args.frozen_good_deterministic
    cfg.device = args.device

    fg_tag = "_fg" if args.frozen_good else ""
    cfg.exp_name = f"{args.run_name}_{cfg.algo}_{args.condition}{fg_tag}_s{args.seed}"

    print("=" * 50)
    print(f"  algo            = {cfg.algo}")
    print(f"  exp_name        = {cfg.exp_name}")
    print(f"  condition       = {cfg.condition}")
    print(f"    message       = {cfg.message_enabled}")
    print(f"    plan_bias     = {cfg.plan_bias_enabled}")
    print(f"    alpha learn   = {cfg.alpha_is_learnable}")
    print(f"  seed            = {cfg.seed}")
    print(f"  device          = {cfg.device}")
    print(f"  total_episodes  = {cfg.total_episodes}")
    print("=" * 50)

    env = make_env(cfg)
    obs, _ = env.reset(seed=cfg.seed)

    print("\n  Agents:")
    for name in env.possible_agents:
        spec = env.get_agent_spec(name)
        print(f"    {name:22s} role={spec.role:10s} "
              f"obs={spec.obs_dim} act={spec.action_dim}")
    print("=" * 50)

    agent = build_agent(cfg, env)
    runner = Runner(cfg, env, agent)

    try:
        runner.run()
    except Exception as e:
        print("\n=== ERROR ===")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
