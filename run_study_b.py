"""
Run Study B (cross-algorithm comparison) and its generalization variant in one go.

Study B       : 3 baselines × 3 seeds, num_good=2, frozen 2-good standard good
Generalization: 4 algorithms × 3 seeds, num_good=3, frozen 3-good standard good

Total: 9 + 12 = 21 training runs.

Usage:
    python run_study_b.py
    python run_study_b.py --only b            # only Study B
    python run_study_b.py --only gen          # only generalization
    python run_study_b.py --seeds 42          # test with one seed first

Assumes standard good checkpoints are at:
  outputs/models/standard_good_mappo_full_coop_s42/final         (2 good)
  outputs/models/standard_good_mappo_full_coop_g3_s42/final      (3 good)
"""

import argparse
import subprocess
import sys
import time
import os


GOOD_2 = "outputs/models/standard_good_mappo_full_coop_s42/final"
GOOD_3 = "outputs/models/standard_good_mappo_full_coop_g3_s42/final"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["b", "gen"], default=None,
                   help="Only run one phase (default: both)")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    p.add_argument("--episodes", type=int, default=None,
                   help="Override total_episodes (e.g. 500 for quick test)")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_cmd(algo, seed, run_name, frozen_good, num_good=None, episodes=None):
    cmd = [sys.executable, "main.py",
           "--algo", algo,
           "--ablation", "full",
           "--coop_reward",
           "--seed", str(seed),
           "--run_name", run_name,
           "--frozen_good", frozen_good]
    if num_good is not None:
        cmd.extend(["--num_good", str(num_good)])
    if episodes is not None:
        cmd.extend(["--total_episodes", str(episodes)])
    return cmd


def main():
    args = parse_args()

    # Check checkpoint existence
    if args.only != "gen" and not os.path.exists(GOOD_2):
        print(f"[ERROR] 2-good standard good checkpoint not found: {GOOD_2}")
        return
    if args.only != "b" and not os.path.exists(GOOD_3):
        print(f"[ERROR] 3-good standard good checkpoint not found: {GOOD_3}")
        return

    runs = []

    # ── Study B: 3 baselines vs 2-good frozen good ───────────────
    if args.only != "gen":
        for seed in args.seeds:
            for algo in ["dqn", "ddpg", "pso"]:
                cmd = build_cmd(algo, seed, "study_b", GOOD_2,
                                num_good=None, episodes=args.episodes)
                runs.append(("study_b", algo, seed, cmd))

    # ── Generalization B: 4 algorithms vs 3-good frozen good ────
    if args.only != "b":
        for seed in args.seeds:
            for algo in ["mappo", "dqn", "ddpg", "pso"]:
                cmd = build_cmd(algo, seed, "study_b_g3", GOOD_3,
                                num_good=3, episodes=args.episodes)
                runs.append(("study_b_g3", algo, seed, cmd))

    total = len(runs)
    print(f"\n{'='*65}")
    print(f"  Total runs: {total}")
    if args.only != "gen":
        print(f"  Study B (2 good):      DQN, DDPG, PSO × {len(args.seeds)} seeds")
    if args.only != "b":
        print(f"  Generalization (3 g):  MAPPO, DQN, DDPG, PSO × {len(args.seeds)} seeds")
    print(f"  Seeds: {args.seeds}")
    if args.episodes:
        print(f"  Episodes: {args.episodes}")
    print(f"{'='*65}\n")

    if args.dry_run:
        for i, (phase, algo, seed, cmd) in enumerate(runs, 1):
            print(f"[{i}/{total}] {phase} {algo} s{seed}")
            print(f"    {' '.join(cmd)}\n")
        return

    overall_start = time.time()
    failed = []

    for i, (phase, algo, seed, cmd) in enumerate(runs, 1):
        print(f"\n{'█'*65}")
        print(f"█ Run {i}/{total}: {phase} | {algo} | seed={seed}")
        print(f"█ Elapsed: {format_time(time.time() - overall_start)}")
        print(f"{'█'*65}\n")

        run_start = time.time()
        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"\n[WARN] Returned code {result.returncode}")
                failed.append((phase, algo, seed))
        except KeyboardInterrupt:
            print(f"\n[ABORTED] Completed {i-1}/{total} runs.")
            return
        except Exception as e:
            print(f"\n[ERROR] {e}")
            failed.append((phase, algo, seed))

        run_elapsed = time.time() - run_start
        print(f"\n  Run took {format_time(run_elapsed)}")

        avg = (time.time() - overall_start) / i
        remaining = avg * (total - i)
        print(f"  Estimated remaining: {format_time(remaining)}")

    # ── Summary ────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"  Total time: {format_time(total_elapsed)}")
    print(f"  Success: {total - len(failed)}/{total}")
    if failed:
        print(f"  Failed:")
        for p, a, s in failed:
            print(f"    - {p} / {a} / seed={s}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
