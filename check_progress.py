"""
Check training progress. Scans outputs/logs/ and reports what's done.

Usage:
    python check_progress.py
"""

import os
import glob
import csv


def last_episode(csv_path):
    """Return the largest episode value in the CSV, or None."""
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            eps = []
            for row in reader:
                try:
                    eps.append(int(row.get("episode", 0)))
                except (ValueError, TypeError):
                    pass
            return max(eps) if eps else None
    except Exception:
        return None


def model_exists(name):
    """Check if a /final or any /ep* checkpoint exists."""
    base = os.path.join("outputs", "models", name)
    if not os.path.isdir(base):
        return False, None
    final = os.path.join(base, "final")
    if os.path.isdir(final):
        return True, "final"
    # look for latest ep checkpoint
    eps = sorted(glob.glob(os.path.join(base, "ep*")))
    if eps:
        return True, os.path.basename(eps[-1])
    return False, None


def main():
    logs = sorted(glob.glob("outputs/logs/*.csv"))
    if not logs:
        print("No CSV logs found in outputs/logs/")
        return

    # Group by run_name prefix
    groups = {}
    for log in logs:
        name = os.path.basename(log).replace(".csv", "")
        # Guess run_name: everything before first "_mappo/_dqn/_ddpg/_pso"
        for algo in ["mappo", "dqn", "ddpg", "pso"]:
            tag = f"_{algo}_"
            if tag in name:
                run_prefix = name.split(tag)[0]
                break
        else:
            run_prefix = name
        groups.setdefault(run_prefix, []).append((name, log))

    print(f"{'Run name':<50s} {'Episodes':>10s} {'Checkpoint':>16s}")
    print("=" * 80)

    for run_name in sorted(groups.keys()):
        print(f"\n[{run_name}]")
        for name, log in sorted(groups[run_name]):
            ep = last_episode(log)
            has_ckpt, ckpt = model_exists(name)

            if ep is None:
                ep_str = "?"
            else:
                ep_str = str(ep)

            ckpt_str = ckpt if has_ckpt else "—"

            # Flag incomplete
            flag = ""
            if ep and ep < 4000:
                flag = "  <-- incomplete"
            elif not has_ckpt:
                flag = "  <-- no checkpoint"

            print(f"  {name:<48s} {ep_str:>10s} {ckpt_str:>16s}{flag}")

    print("\n" + "=" * 80)

    # Summary: expected experiments
    print("\nExpected experiments checklist:\n")

    checklist = []
    # Standard goods
    checklist.append(("standard_good_mappo_full_coop_s42", "Standard good (2 good)"))
    checklist.append(("standard_good_mappo_full_coop_g3_s42", "Standard good (3 good)"))
    # Study A: 4 ablations x 3 seeds
    for ab in ["full", "no_comm", "no_leader_pos", "blind"]:
        for s in [42, 123, 2024]:
            checklist.append((f"study_a_mappo_{ab}_coop_fg_s{s}",
                              f"Study A: {ab} s{s}"))
    # Study B: 3 baselines x 3 seeds, 2 good
    for algo in ["dqn", "ddpg", "pso"]:
        for s in [42, 123, 2024]:
            checklist.append((f"study_b_{algo}_full_coop_fg_s{s}",
                              f"Study B: {algo} s{s}"))
    # Gen B: 4 algos x 3 seeds, 3 good
    for algo in ["mappo", "dqn", "ddpg", "pso"]:
        for s in [42, 123, 2024]:
            checklist.append((f"study_b_g3_{algo}_full_coop_fg_g3_s{s}",
                              f"Gen B: {algo} s{s}"))

    done = 0
    partial = 0
    missing = 0
    for name, label in checklist:
        log_path = f"outputs/logs/{name}.csv"
        if not os.path.exists(log_path):
            status = "[ ] MISSING"
            missing += 1
        else:
            ep = last_episode(log_path)
            if ep and ep >= 4000:
                status = "[X] DONE"
                done += 1
            else:
                status = f"[~] PARTIAL (ep {ep})"
                partial += 1
        print(f"  {status:<20s} {label}")

    total = len(checklist)
    print(f"\n  Done: {done}/{total}  |  Partial: {partial}  |  Missing: {missing}")


if __name__ == "__main__":
    main()
