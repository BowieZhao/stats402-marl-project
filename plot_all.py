"""
Plot Study A: 4-way ablation comparison of MAPPO, averaged across 3 seeds.
Produces three figures:
  1. ablation_curves.png   training reward curves (mean +/- std across seeds)
  2. ablation_eval.png     eval reward over training
  3. ablation_bars.png     final average reward bar chart
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_DIR = "outputs/logs"
OUT_DIR = "outputs"
RUN_PREFIX = "study_a"

ABLATIONS = ["full", "no_comm", "no_leader_pos", "blind"]
SEEDS = [42, 123, 2024]

COLORS = {
    "full": "#2196F3",
    "no_comm": "#FF9800",
    "no_leader_pos": "#4CAF50",
    "blind": "#F44336",
}
LABELS = {
    "full": "full (all info)",
    "no_comm": "no_comm (zero 4-bit msg)",
    "no_leader_pos": "no_leader_pos (zero pos/vel)",
    "blind": "blind (both zeroed)",
}


def load_run(ablation, seed):
    pattern = os.path.join(
        LOG_DIR, f"{RUN_PREFIX}_mappo_{ablation}_coop_fg_s{seed}.csv"
    )
    files = glob.glob(pattern)
    if not files:
        print(f"  [MISS] {pattern}")
        return None, None

    df = pd.read_csv(files[0])

    # Coerce numeric columns (ignore columns that don't exist)
    for col in ["episode", "adv_reward", "good_reward",
                "eval_adv_mean", "eval_adv_std",
                "eval_good_mean", "eval_good_std"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Split train vs eval
    # Strategy: eval rows have eval_adv_mean populated; train rows have adv_reward populated.
    if "eval_adv_mean" in df.columns:
        eval_df = df[df["eval_adv_mean"].notna()].copy()
    else:
        eval_df = df.iloc[0:0].copy()  # empty

    if "adv_reward" in df.columns:
        train_df = df[df["adv_reward"].notna()].copy()
    else:
        train_df = df.iloc[0:0].copy()

    return train_df, eval_df


runs = {}
for ab in ABLATIONS:
    runs[ab] = []
    for seed in SEEDS:
        train_df, eval_df = load_run(ab, seed)
        if train_df is not None and len(train_df) > 0:
            runs[ab].append((train_df, eval_df))
            print(f"  [OK] {ab} s{seed}: {len(train_df)} train, {len(eval_df)} eval rows")


def plot_training_curves():
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    window = 50

    for ab in ABLATIONS:
        if not runs[ab]:
            continue
        color = COLORS[ab]

        adv_curves, good_curves = [], []
        ep_grid = None
        for train_df, _ in runs[ab]:
            ep = train_df["episode"].values.astype(float)
            adv = train_df["adv_reward"].values.astype(float)
            good = train_df["good_reward"].values.astype(float)
            adv_s = pd.Series(adv).rolling(window, min_periods=1).mean().values
            good_s = pd.Series(good).rolling(window, min_periods=1).mean().values
            adv_curves.append((ep, adv_s))
            good_curves.append((ep, good_s))
            if ep_grid is None or len(ep) > len(ep_grid):
                ep_grid = ep

        adv_mat = np.array([np.interp(ep_grid, ep, y) for (ep, y) in adv_curves])
        good_mat = np.array([np.interp(ep_grid, ep, y) for (ep, y) in good_curves])

        axes[0].plot(ep_grid, adv_mat.mean(axis=0), color=color, linewidth=2.5,
                     label=LABELS[ab])
        axes[0].fill_between(ep_grid,
                             adv_mat.mean(axis=0) - adv_mat.std(axis=0),
                             adv_mat.mean(axis=0) + adv_mat.std(axis=0),
                             color=color, alpha=0.15)

        axes[1].plot(ep_grid, good_mat.mean(axis=0), color=color, linewidth=2.5,
                     label=LABELS[ab])
        axes[1].fill_between(ep_grid,
                             good_mat.mean(axis=0) - good_mat.std(axis=0),
                             good_mat.mean(axis=0) + good_mat.std(axis=0),
                             color=color, alpha=0.15)

    axes[0].set_ylabel("Adversary Team Reward", fontsize=12)
    axes[0].set_title("Study A: MAPPO Ablation - Training Reward (mean +/- std, 3 seeds)",
                      fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=11, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Good Agent Reward", fontsize=12)
    axes[1].set_xlabel("Episode", fontsize=12)
    axes[1].legend(fontsize=10, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ablation_curves.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[Saved] {out}")
    plt.close()


def plot_eval_curves():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    plotted = 0
    for ab in ABLATIONS:
        if not runs[ab]:
            continue
        color = COLORS[ab]

        all_evals = []
        ep_grid = None
        for _, eval_df in runs[ab]:
            if len(eval_df) == 0 or "eval_adv_mean" not in eval_df.columns:
                continue
            ep = eval_df["episode"].values.astype(float)
            vals = eval_df["eval_adv_mean"].values.astype(float)
            all_evals.append((ep, vals))
            if ep_grid is None or len(ep) > len(ep_grid):
                ep_grid = ep

        if not all_evals:
            continue

        mat = np.array([np.interp(ep_grid, ep, vals) for (ep, vals) in all_evals])
        ax.plot(ep_grid, mat.mean(axis=0), color=color, linewidth=2.5,
                marker="o", markersize=5, label=LABELS[ab])
        ax.fill_between(ep_grid,
                        mat.mean(axis=0) - mat.std(axis=0),
                        mat.mean(axis=0) + mat.std(axis=0),
                        color=color, alpha=0.15)
        plotted += 1

    if plotted == 0:
        print("[WARN] No eval data found; skipping ablation_eval.png")
        plt.close()
        return

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Eval Adversary Reward", fontsize=12)
    ax.set_title("Study A: Evaluation Reward (deterministic, mean +/- std, 3 seeds)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ablation_eval.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[Saved] {out}")
    plt.close()


def plot_final_bars():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    labels, means, stds, colors_used = [], [], [], []

    for ab in ABLATIONS:
        if not runs[ab]:
            continue
        finals = [t.tail(25)["adv_reward"].mean() for t, _ in runs[ab]]
        finals = np.array(finals)
        labels.append(LABELS[ab])
        means.append(finals.mean())
        stds.append(finals.std())
        colors_used.append(COLORS[ab])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=10, color=colors_used,
                  alpha=0.88, edgecolor="black", linewidth=1)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 3,
                f"{m:.1f}+/-{s:.1f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Final Adv Reward (last 500 ep, mean +/- std, 3 seeds)", fontsize=11)
    ax.set_title("Study A: Ablation - Final Performance",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ablation_bars.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[Saved] {out}")
    plt.close()


def print_summary():
    print("\n" + "=" * 75)
    print(f"{'Condition':<25s} {'Final train adv':>22s} {'Final eval adv':>22s}")
    print("=" * 75)
    for ab in ABLATIONS:
        if not runs[ab]:
            print(f"  {ab:<23s}  NO DATA")
            continue
        finals = np.array([t.tail(25)["adv_reward"].mean() for t, _ in runs[ab]])
        eval_finals = np.array([
            e.tail(5)["eval_adv_mean"].mean() for _, e in runs[ab]
            if len(e) > 0 and "eval_adv_mean" in e.columns
        ])
        eval_str = (f"{eval_finals.mean():>7.2f} +/- {eval_finals.std():<6.2f}"
                    if len(eval_finals) > 0 else "N/A")
        print(f"  {ab:<23s}  {finals.mean():>7.2f} +/- {finals.std():<6.2f}   {eval_str}")
    print("=" * 75)

    print("\nKey comparisons (Welch t-test):")
    for (a, b) in [("full", "no_comm"),
                   ("no_leader_pos", "blind"),
                   ("full", "blind")]:
        if runs[a] and runs[b]:
            a_vals = np.array([t.tail(25)["adv_reward"].mean() for t, _ in runs[a]])
            b_vals = np.array([t.tail(25)["adv_reward"].mean() for t, _ in runs[b]])
            diff = a_vals.mean() - b_vals.mean()
            try:
                from scipy import stats as sps
                t, p = sps.ttest_ind(a_vals, b_vals, equal_var=False)
                sig = " ***" if p < 0.01 else (" **" if p < 0.05 else (" *" if p < 0.1 else ""))
                print(f"  {a:<16s} vs {b:<16s}: diff = {diff:+7.2f}   "
                      f"t = {t:+.2f}  p = {p:.4f}{sig}")
            except ImportError:
                print(f"  {a:<16s} vs {b:<16s}: diff = {diff:+7.2f}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    any_data = any(runs[ab] for ab in ABLATIONS)
    if not any_data:
        print("No data found.")
        import sys; sys.exit(1)

    plot_training_curves()
    plot_eval_curves()
    plot_final_bars()
    print_summary()

    print("\nDone. Check outputs/ for ablation_curves.png, ablation_eval.png, ablation_bars.png")
