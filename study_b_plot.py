"""
Study B: MAPPO (Ours) vs DQN/DDPG/PSO algorithm comparison.
Uses hardcoded filenames matching the actual logs directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────
LOG_DIR = r"C:\Users\86135\Desktop\Stats402\final_version\marl_comm_v2\outputs\logs"

FILE_MAPPO = "NEW_PPO_KING_mappo_E1_full_s42.csv"
FILE_DQN   = "exp_dqn_E1_s42_mappo_E1_full_s42.csv"
FILE_DDPG  = "exp_ddpg_E1_s42_mappo_E1_full_s42.csv"
FILE_PSO   = "exp_PSO_E1_full_s42.csv"

SMOOTH_WINDOW = 50
EP_CUTOFF = 4000
TAIL_CUTOFF = None

OUT_PATH = r"C:\Users\86135\Desktop\Stats402\final_version\marl_comm_v2\study_b_algo_comparison.png"


def load_smooth(filename, window, ep_cutoff, tail_cutoff):
    full_path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Not found: {full_path}")
    df = pd.read_csv(full_path)
    y_col = "adv_return" if "adv_return" in df.columns else "adv"
    x_col = "episode" if "episode" in df.columns else df.columns[0]

    if ep_cutoff:
        df = df[df[x_col] <= ep_cutoff].copy()
    if tail_cutoff:
        df = df[df[x_col] <= tail_cutoff].copy()

    df["smoothed"] = df[y_col].rolling(window=window, min_periods=1).mean()
    return df, x_col, y_col


# ─── Load ──────────────────────────────────────────────────
print(f"Loading from: {LOG_DIR}\n")
print(f"  MAPPO: {FILE_MAPPO}")
mappo_df, x_col, y_col = load_smooth(FILE_MAPPO, SMOOTH_WINDOW, EP_CUTOFF, TAIL_CUTOFF)
print(f"  DQN:   {FILE_DQN}")
dqn_df, _, _ = load_smooth(FILE_DQN, SMOOTH_WINDOW, EP_CUTOFF, TAIL_CUTOFF)
print(f"  DDPG:  {FILE_DDPG}")
ddpg_df, _, _ = load_smooth(FILE_DDPG, SMOOTH_WINDOW, EP_CUTOFF, TAIL_CUTOFF)
print(f"  PSO:   {FILE_PSO}")
pso_df, _, _ = load_smooth(FILE_PSO, SMOOTH_WINDOW, EP_CUTOFF, TAIL_CUTOFF)

# ─── Plot ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"width_ratios": [3, 1]})

runs = [
    (mappo_df, "MAPPO (Ours)",   "#d62728", "-",  10),
    (dqn_df,   "DQN Baseline",   "#2ca02c", "-",   3),
    (ddpg_df,  "DDPG Baseline",  "#1f77b4", "-",   2),
    (pso_df,   "PSO Baseline",   "#ff7f0e", "-",   1),
]

for df, label, color, ls, z in runs:
    ax1.plot(df[x_col], df[y_col], color=color, alpha=0.10, linewidth=0.8, zorder=z)
    lw = 2.8 if "MAPPO" in label else 2.2
    ax1.plot(df[x_col], df["smoothed"], color=color, linestyle=ls,
             linewidth=lw, label=label, zorder=z + 5)

ax1.set_title("Algorithm Comparison in Communicative MARL (Study B)",
              fontsize=14, fontweight="bold")
ax1.set_xlabel("Training Episodes", fontsize=12)
ax1.set_ylabel("Team Reward (smoothed, window=50)", fontsize=12)
ax1.legend(loc="upper left", fontsize=11, framealpha=0.95)
ax1.grid(True, linestyle=":", alpha=0.6)

# Right: peak performance
peaks = []
for df, label, color, _, _ in runs:
    peak = float(df["smoothed"].max())
    short = label.split(" ")[0]
    peaks.append((short, peak, color))

bar_labels = [p[0] for p in peaks]
bar_values = [p[1] for p in peaks]
bar_colors = [p[2] for p in peaks]

bars = ax2.bar(bar_labels, bar_values, color=bar_colors, alpha=0.85, edgecolor="black")
for bar, val in zip(bars, bar_values):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 2,
             f"{val:.0f}", ha="center", fontsize=11, fontweight="bold")

ax2.set_title("Peak Performance", fontsize=14, fontweight="bold")
ax2.set_ylabel("Max smoothed reward", fontsize=12)
ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"\nSaved: {OUT_PATH}")

print("\n" + "=" * 50)
print("Summary (peak smoothed reward):")
for label, val, _ in peaks:
    print(f"  {label}: {val:.2f}")
print("=" * 50)

plt.show()
