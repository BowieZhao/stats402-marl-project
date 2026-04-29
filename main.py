"""
Main entry point for FINAL design experiments.
Runs MAPPO with one of three conditions: E1_full / E2_no_comm / E3_no_alpha.
"""

import argparse
import os
import sys
from config import Config
from envs import make_env
from algorithms.mappo import MAPPOAgent
from experiment import Runner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="E1_full",
                   choices=["E1_full", "E2_no_comm", "E3_no_alpha"],
                   help="Experiment condition")
    p.add_argument("--total_episodes", type=int, default=4000)
    p.add_argument("--max_cycles", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default="exp")
    p.add_argument("--num_good", type=int, default=2)
    p.add_argument("--num_forests", type=int, default=2)
    p.add_argument("--frozen_good", default=None,
                   help="Path to frozen good policy directory")
    p.add_argument("--frozen_good_deterministic", action="store_true")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.algo = "mappo"
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

    # Build experiment name
    fg_tag = "_fg" if args.frozen_good else ""
    cfg.exp_name = f"{args.run_name}_mappo_{args.condition}{fg_tag}_s{args.seed}"

    print("=" * 50)
    print(f"  algo            = {cfg.algo}")
    print(f"  exp_name        = {cfg.exp_name}")
    print(f"  condition       = {cfg.condition}")
    print(f"    message       = {cfg.message_enabled}")
    print(f"    plan_bias     = {cfg.plan_bias_enabled}")
    print(f"    alpha learn   = {cfg.alpha_is_learnable}")
    print(f"  seed            = {cfg.seed}")
    print(f"  device          = {cfg.device}")
    print(f"  num_good        = {cfg.num_good}")
    print(f"  num_forests     = {cfg.num_forests}")
    print(f"  total_episodes  = {cfg.total_episodes}")
    print(f"  frozen_good     = {cfg.frozen_good_path}")
    print("=" * 50)

    env = make_env(cfg)
    obs, _ = env.reset(seed=cfg.seed)

    print("\n  Agents:")
    for name in env.possible_agents:
        spec = env.get_agent_spec(name)
        print(f"    {name:22s} role={spec.role:10s} "
              f"obs={spec.obs_dim} act={spec.action_dim} "
              f"type={spec.action_type}")
    print("=" * 50)

    agent = MAPPOAgent(cfg, env)

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
