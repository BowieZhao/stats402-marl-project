"""
Batch runner for the FINAL design experiments.

Runs E1_full / E2_no_comm / E3_no_alpha × 3 seeds = 9 runs.
Total time: ~2.5 hours on CPU.

Usage:
    python run_experiments.py --frozen_good outputs/models/standard_good_mappo_full_coop_s42/final
"""

import argparse
import subprocess
import sys
import time
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+",
                   default=["E1_full", "E2_no_comm", "E3_no_alpha"],
                   help="Which conditions to run")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--total_episodes", type=int, default=4000)
    p.add_argument("--run_name", default="final")
    p.add_argument("--frozen_good", required=True,
                   help="Path to frozen good checkpoint dir")
    p.add_argument("--num_good", type=int, default=2)
    p.add_argument("--num_forests", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    runs = [(c, s) for c in args.conditions for s in args.seeds]
    n = len(runs)

    print()
    print("=" * 60)
    print(f"  Final design: {len(args.conditions)} conditions × "
          f"{len(args.seeds)} seeds = {n} runs")
    print(f"  Conditions: {args.conditions}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Run name:   {args.run_name}")
    print(f"  Frozen good: {args.frozen_good}")
    print("=" * 60)

    overall_start = time.time()

    for i, (condition, seed) in enumerate(runs, 1):
        print()
        print("█" * 60)
        print(f"█ Run {i}/{n}: condition={condition}  seed={seed}")
        elapsed = time.time() - overall_start
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        print(f"█ Elapsed so far: {h:02d}:{m:02d}:{s:02d}")
        print("█" * 60)
        print()

        cmd = [
            sys.executable, "main.py",
            "--condition", condition,
            "--seed", str(seed),
            "--run_name", args.run_name,
            "--total_episodes", str(args.total_episodes),
            "--frozen_good", args.frozen_good,
            "--num_good", str(args.num_good),
            "--num_forests", str(args.num_forests),
        ]
        run_start = time.time()
        result = subprocess.run(cmd)

        run_dur = time.time() - run_start
        rh = int(run_dur // 3600)
        rm = int((run_dur % 3600) // 60)
        rs = int(run_dur % 60)
        print()
        print(f"  Run completed in {rh:02d}:{rm:02d}:{rs:02d}")

        if result.returncode != 0:
            print(f"[WARN] Run returned code {result.returncode}")

        remaining_runs = n - i
        avg_per = (time.time() - overall_start) / i
        est = remaining_runs * avg_per
        eh = int(est // 3600)
        em = int((est % 3600) // 60)
        es = int(est % 60)
        print(f"  Estimated remaining: {eh:02d}:{em:02d}:{es:02d}")

    total = time.time() - overall_start
    th = int(total // 3600)
    tm = int((total % 3600) // 60)
    ts = int(total % 60)
    print()
    print("=" * 60)
    print(f"All runs done. Total time: {th:02d}:{tm:02d}:{ts:02d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
