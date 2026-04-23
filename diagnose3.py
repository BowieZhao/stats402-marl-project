"""
Final diagnosis after running 12-experiment ablation.

1. Move old (num_forests=2) logs out of the way
2. Re-plot with only num_forests=4 runs
3. Verify the actual occlusion rate seen by the trained policy
   (not random policy as in diagnose2.py)
"""

import os
import shutil
import glob
import numpy as np

from config import Config
from envs import WorldCommEnv, is_normal_adversary


# ═══════════════════════════════════════════════════════════
# Step 1: Move old logs aside
# ═══════════════════════════════════════════════════════════
def cleanup_old_logs():
    print("=" * 60)
    print("STEP 1: Separating old (num_forests=2) and new (num_forests=4) logs")
    print("=" * 60)

    old_dir = "outputs/logs_old_nf2"
    os.makedirs(old_dir, exist_ok=True)

    moved = 0
    for f in glob.glob("outputs/logs/*.csv"):
        # New runs have "swc_f4_" prefix; old runs have "swc_mappo_"
        basename = os.path.basename(f)
        if basename.startswith("swc_mappo_") and "f4" not in basename:
            dst = os.path.join(old_dir, basename)
            shutil.move(f, dst)
            print(f"  moved {basename} → logs_old_nf2/")
            moved += 1

    # Also move JSON configs
    for f in glob.glob("outputs/logs/*.json"):
        basename = os.path.basename(f)
        if basename.startswith("swc_mappo_") and "f4" not in basename:
            dst = os.path.join(old_dir, basename)
            shutil.move(f, dst)
            moved += 1

    print(f"\n  Moved {moved} old files to outputs/logs_old_nf2/")
    print(f"  Current outputs/logs/ contains only num_forests=4 runs.")


# ═══════════════════════════════════════════════════════════
# Step 2: Measure actual occlusion under trained policy
# ═══════════════════════════════════════════════════════════
def measure_trained_occlusion():
    print("\n" + "=" * 60)
    print("STEP 2: Measuring good-agent occlusion with TRAINED policy")
    print("=" * 60)

    from algorithms.mappo import MAPPOAgent

    ckpt_path = "outputs/models/swc_f4_mappo_full_s42/final"
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}")
        print(f"  Listing available:")
        for d in glob.glob("outputs/models/swc_f4*"):
            print(f"    {d}")
        return

    config = Config(num_forests=4, ablation_mode="full")
    env = WorldCommEnv(config)
    env.reset(seed=42)

    agent = MAPPOAgent(config, env)
    agent.load(ckpt_path)

    total_steps = 0
    g0_occluded = 0
    g1_occluded = 0
    any_occluded = 0
    both_occluded = 0

    for ep in range(30):
        obs, _ = env.reset(seed=ep + 1000)
        while env.agents:
            actions = agent.select_actions(obs, env, explore=False)
            active = {a: actions[a] for a in env.agents if a in actions}
            obs, _, _, _, _ = env.step(active)

            if "adversary_0" in obs:
                ob = obs["adversary_0"]
                g0 = ob[20:22]
                g1 = ob[22:24]
                o0 = np.allclose(g0, 0, atol=1e-6)
                o1 = np.allclose(g1, 0, atol=1e-6)
                if o0: g0_occluded += 1
                if o1: g1_occluded += 1
                if o0 or o1: any_occluded += 1
                if o0 and o1: both_occluded += 1
                total_steps += 1

    if total_steps == 0:
        print("  No data collected.")
        return

    print(f"\n  With TRAINED policy over {total_steps} steps:")
    print(f"    good_0 occluded: {g0_occluded} ({100*g0_occluded/total_steps:.1f}%)")
    print(f"    good_1 occluded: {g1_occluded} ({100*g1_occluded/total_steps:.1f}%)")
    print(f"    any occluded:    {any_occluded} ({100*any_occluded/total_steps:.1f}%)")
    print(f"    both occluded:   {both_occluded} ({100*both_occluded/total_steps:.1f}%)")
    print(f"\n  Compare to random policy: 69% any-occluded (from diagnose2.py)")

    if any_occluded / total_steps < 0.3:
        print("\n  ⚠ Trained policy FINDS good agents despite forests!")
        print("  Adversaries likely learned to actively drive good agents out of forests,")
        print("  or to approach forests to force good agents into visible territory.")
        print("  This explains why ablations don't matter: effective occlusion is minimal.")
    else:
        print("\n  ✓ Occlusion is substantial. Other reasons must explain the null result.")

    env.close()


# ═══════════════════════════════════════════════════════════
# Step 3: Check eval data quality
# ═══════════════════════════════════════════════════════════
def check_eval_data():
    print("\n" + "=" * 60)
    print("STEP 3: Checking eval reward across conditions")
    print("=" * 60)

    import pandas as pd

    files = sorted(glob.glob("outputs/logs/swc_f4_*.csv"))
    if not files:
        print("  No num_forests=4 CSVs found.")
        return

    results = {}  # {ablation: [eval_rewards across seeds]}

    for f in files:
        df = pd.read_csv(f)
        # Extract ablation from filename
        basename = os.path.basename(f)
        # swc_f4_mappo_full_s42.csv
        parts = basename.replace(".csv", "").split("_")
        # Find ablation (between 'mappo' and 's<num>')
        try:
            mappo_idx = parts.index("mappo")
            seed_idx = next(i for i, p in enumerate(parts) if p.startswith("s") and p[1:].isdigit())
            ablation = "_".join(parts[mappo_idx+1:seed_idx])
            seed = parts[seed_idx]
        except (ValueError, StopIteration):
            continue

        # Get eval rows
        if "type" in df.columns:
            evals = df[df["type"] == "eval"].copy()
        else:
            continue

        if evals.empty or "eval_adv_mean" not in evals.columns:
            continue

        evals["eval_adv_mean"] = pd.to_numeric(evals["eval_adv_mean"], errors="coerce")
        evals = evals.dropna(subset=["eval_adv_mean"])

        # Take last 5 eval points (i.e., last ~500 episodes of eval)
        last_evals = evals.tail(5)["eval_adv_mean"].values

        if ablation not in results:
            results[ablation] = []
        results[ablation].extend(last_evals)

    print(f"\n  Final eval rewards (last 500 episodes, aggregated across seeds):")
    print(f"  {'Condition':<18s} {'Mean':>8s} {'Std':>8s} {'N':>4s}")
    print(f"  {'-'*40}")
    for abl in ["full", "no_comm", "no_leader_pos", "blind"]:
        if abl in results:
            vals = np.array(results[abl])
            print(f"  {abl:<18s} {vals.mean():>8.2f} {vals.std():>8.2f} {len(vals):>4d}")

    # Key t-test
    if "no_leader_pos" in results and "blind" in results:
        from scipy import stats
        try:
            nlp = np.array(results["no_leader_pos"])
            bld = np.array(results["blind"])
            t, p = stats.ttest_ind(nlp, bld, equal_var=False)
            print(f"\n  Welch t-test: no_leader_pos vs blind")
            print(f"    t = {t:.3f}, p = {p:.4f}")
            if p < 0.05:
                print(f"    → SIGNIFICANT difference")
            else:
                print(f"    → NOT significant (p > 0.05)")
        except ImportError:
            print("\n  (install scipy for t-test)")


if __name__ == "__main__":
    cleanup_old_logs()
    measure_trained_occlusion()
    check_eval_data()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Re-run plot_all.py now that old logs are separated:
   python plot_all.py

2. Look at the 'trained occlusion rate' above:
   - If < 30%: adversaries learned to avoid forests → true occlusion is low
     → need to change env further (e.g., larger forests, fewer food,
        bigger map so agents can't just chase everywhere)
   - If > 50%: occlusion is real, but information still isn't critical
     → means task is still solvable without leader info
     → accept the null result, reframe paper

3. Look at t-test p-value:
   - If p < 0.05: there IS a small effect hidden in the noise
   - If p > 0.3: genuinely no effect, stop trying to force it
""")
