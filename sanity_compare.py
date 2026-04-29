"""
Quick sanity comparison: run E1 and E2 back-to-back, 500 episodes each.
Total time: ~5-6 minutes.

Usage:
    python sanity_compare.py
"""

import subprocess
import sys
import time

EPISODES = 500
SEED = 42
RUN_NAME = "sanity"

runs = [
    ("E1_full", "Comm ON: message + plan_bias + alpha learnable"),
    ("E2_no_comm", "Comm OFF: message blocked, plan_bias off"),
]

print("=" * 60)
print("Sanity comparison: E1 vs E2")
print(f"Episodes per run: {EPISODES}")
print(f"Seed: {SEED}")
print("=" * 60)

start_total = time.time()

for i, (cond, desc) in enumerate(runs, 1):
    print()
    print("█" * 60)
    print(f"█ Run {i}/{len(runs)}: {cond}")
    print(f"█ {desc}")
    print("█" * 60)
    print()

    cmd = [
        sys.executable, "main.py",
        "--condition", cond,
        "--seed", str(SEED),
        "--run_name", RUN_NAME,
        "--total_episodes", str(EPISODES),
    ]
    t0 = time.time()
    result = subprocess.run(cmd)
    dur = time.time() - t0
    print()
    print(f"  Run done in {dur/60:.1f} min")

total = time.time() - start_total
print()
print("=" * 60)
print(f"All done. Total time: {total/60:.1f} min")
print()
print("Compare with:")
print(f"  python visualize.py --checkpoint outputs/models/{RUN_NAME}_mappo_E1_full_s{SEED}/final --breakdown --n_episodes 5")
print(f"  python visualize.py --checkpoint outputs/models/{RUN_NAME}_mappo_E2_no_comm_s{SEED}/final --breakdown --n_episodes 5")
print("=" * 60)
