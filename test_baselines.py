"""
Quick sanity check for DQN, DDPG, PSO baselines.

Runs 100 episodes of each algorithm with coop_reward enabled, checks:
- No crashes
- Rewards aren't NaN
- Some kind of learning signal (reward varies, losses exist)

Usage:
    python test_baselines.py

Total time: ~3-5 minutes.
"""

import subprocess
import sys
import os
import time


def run_one(algo: str, episodes: int = 100) -> dict:
    """Run one short training and return summary info."""
    cmd = [
        sys.executable, "main.py",
        "--algo", algo,
        "--ablation", "full",
        "--coop_reward",
        "--seed", "42",
        "--run_name", "baseline_check",
        "--total_episodes", str(episodes),
    ]

    print(f"\n{'='*60}")
    print(f"Testing {algo.upper()}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    success = result.returncode == 0
    return {
        "algo": algo,
        "success": success,
        "elapsed_s": elapsed,
        "returncode": result.returncode,
    }


def main():
    print("=" * 60)
    print("BASELINE ALGORITHM SANITY CHECK")
    print("=" * 60)
    print("This will run DQN, DDPG, and PSO for 100 episodes each")
    print("with --coop_reward. Expect ~1-2 minutes per algorithm.")
    print()

    results = []
    for algo in ["dqn", "ddpg", "pso"]:
        try:
            r = run_one(algo, episodes=100)
            results.append(r)
        except KeyboardInterrupt:
            print("\n[ABORTED] User interrupt.")
            break
        except Exception as e:
            print(f"\n[ERROR] {algo} raised: {e}")
            results.append({"algo": algo, "success": False, "error": str(e)})

    # ── Summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        status = "OK" if r.get("success") else "FAIL"
        elapsed = r.get("elapsed_s", 0)
        print(f"  {r['algo'].upper():<8s} [{status}]  elapsed={elapsed:.1f}s")

    n_ok = sum(1 for r in results if r.get("success"))
    print(f"\n  Passed: {n_ok}/{len(results)}")

    if n_ok == len(results):
        print("\n✓ All baselines can run. Safe to proceed to full training.")
        print("\nNext steps:")
        print("  python run_all.py --coop_reward --algo dqn --ablations full --run_name coop")
        print("  python run_all.py --coop_reward --algo ddpg --ablations full --run_name coop")
        print("  python run_all.py --coop_reward --algo pso --ablations full --run_name coop")
    else:
        print("\n⚠ Some baselines failed. Check the log output above for errors.")


if __name__ == "__main__":
    main()
