"""
Run all ablation experiments sequentially.

Usage:
    python run_all.py
    python run_all.py --seeds 42 123 2024
    python run_all.py --ablations full blind    # subset
    python run_all.py --episodes 3000           # override episode count
"""

import argparse
import subprocess
import sys
import time
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024],
                   help="Random seeds to run")
    p.add_argument("--ablations", nargs="+", default=["full", "no_comm", "no_leader_pos", "blind"],
                   help="Ablation modes to run")
    p.add_argument("--episodes", type=int, default=None,
                   help="Override total_episodes (default: use config.py value)")
    p.add_argument("--run_name", type=str, default="swc_f4",
                   help="Experiment run name prefix")
    p.add_argument("--coop_reward", action="store_true",
                   help="Enable cooperative reward shaping for all runs")
    p.add_argument("--num_good", type=int, default=None,
                   help="Override num_good for all runs (2 or 3)")
    p.add_argument("--algo", type=str, default="mappo",
                   choices=["mappo", "dqn", "ddpg", "pso"],
                   help="Algorithm to run")
    p.add_argument("--frozen_good", type=str, default=None,
                   help="Path to MAPPO checkpoint dir (frozen good policy)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    return p.parse_args()


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    args = parse_args()

    # Build command list
    runs = []
    for seed in args.seeds:
        for ablation in args.ablations:
            cmd = [sys.executable, "main.py",
                   "--algo", args.algo,
                   "--ablation", ablation,
                   "--seed", str(seed),
                   "--run_name", args.run_name]
            if args.episodes is not None:
                cmd.extend(["--total_episodes", str(args.episodes)])
            if args.coop_reward:
                cmd.append("--coop_reward")
            if args.num_good is not None:
                cmd.extend(["--num_good", str(args.num_good)])
            if args.frozen_good is not None:
                cmd.extend(["--frozen_good", args.frozen_good])
            runs.append((ablation, seed, cmd))

    total = len(runs)
    print(f"\n{'='*60}")
    print(f"  Ablation study: {len(args.ablations)} ablations × {len(args.seeds)} seeds = {total} runs")
    print(f"  Ablations: {args.ablations}")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Run name:  {args.run_name}")
    if args.episodes:
        print(f"  Episodes:  {args.episodes}")
    print(f"{'='*60}\n")

    if args.dry_run:
        for i, (abl, seed, cmd) in enumerate(runs, 1):
            print(f"[{i}/{total}] {' '.join(cmd)}")
        return

    overall_start = time.time()
    failed = []

    for i, (ablation, seed, cmd) in enumerate(runs, 1):
        print(f"\n{'█'*60}")
        print(f"█ Run {i}/{total}: ablation={ablation}  seed={seed}")
        print(f"█ Elapsed so far: {format_time(time.time() - overall_start)}")
        print(f"{'█'*60}\n")

        run_start = time.time()
        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"\n[WARN] Run returned code {result.returncode}")
                failed.append((ablation, seed))
        except KeyboardInterrupt:
            print("\n[ABORTED] User interrupted.")
            print(f"Completed: {i-1}/{total}")
            print(f"Failed: {failed}")
            return
        except Exception as e:
            print(f"\n[ERROR] {e}")
            failed.append((ablation, seed))

        run_elapsed = time.time() - run_start
        print(f"\n  Run completed in {format_time(run_elapsed)}")

        # Estimate remaining time
        avg_time = (time.time() - overall_start) / i
        remaining = avg_time * (total - i)
        print(f"  Estimated remaining: {format_time(remaining)}")

    # ── Summary ────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  ALL RUNS COMPLETE")
    print(f"  Total time: {format_time(total_elapsed)}")
    print(f"  Successful: {total - len(failed)}/{total}")
    if failed:
        print(f"  Failed:")
        for abl, seed in failed:
            print(f"    - ablation={abl} seed={seed}")
    print(f"{'='*60}\n")
    print("Next step:")
    print(f"  python plot_all.py")


if __name__ == "__main__":
    main()
