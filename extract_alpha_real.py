"""
Extract REAL alpha distribution from MAPPO E1 checkpoints.

Unlike the previous plot_real_alpha.py which only read alpha_head.bias (ignoring
the W·h term), this script:
  1. loads each checkpoint
  2. runs N evaluation episodes through it
  3. records the actual alpha output (sigmoid(W·h + b)) at every step
  4. plots:
     (a) per-checkpoint alpha distribution (boxplot)
     (b) per-checkpoint mean alpha trajectory
     (c) within-episode alpha variation at the final checkpoint

Usage:
    python extract_alpha_real.py --run_dir outputs/models/final_mappo_E1_full_s42
"""

import argparse
import os
import re
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from config import Config
from envs import make_env, is_normal_adversary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="Path to outputs/models/<run_name> directory")
    p.add_argument("--n_episodes", type=int, default=20,
                   help="Number of evaluation episodes per checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_prefix", default="alpha_real",
                   help="Output PNG prefix")
    return p.parse_args()


def get_ep_num(folder_path):
    name = os.path.basename(folder_path)
    if name == "final":
        return None  # we'll resolve later
    m = re.search(r"\d+", name)
    return int(m.group(0)) if m else None


def find_checkpoints(run_dir):
    """Return list of (ep_num, folder_path) sorted by ep_num."""
    eps = glob.glob(os.path.join(run_dir, "ep*"))
    finals = glob.glob(os.path.join(run_dir, "final"))

    out = []
    max_ep = 0
    for f in eps:
        n = get_ep_num(f)
        if n is not None:
            out.append((n, f))
            max_ep = max(max_ep, n)
    for f in finals:
        if max_ep == 0:
            out.append((0, f))
        else:
            print("Skipping final checkpoint (ep4000 already exists)")

    out.sort(key=lambda x: x[0])
    return out


def load_actor_state(folder, role_name="adversary"):
    """Find the adversary actor state dict."""
    candidates = [
        os.path.join(folder, f"{role_name}.pt"),
        os.path.join(folder, f"{role_name}_actor.pt"),
        os.path.join(folder, f"actor_{role_name}.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            data = torch.load(c, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                if "model" in data: return data["model"]
                if "actor" in data: return data["actor"]
                if "state_dict" in data: return data["state_dict"]
                return data
            return data
    return None


def build_actor_from_state(state_dict, obs_dim, action_dim):
    """Reconstruct AdversaryAlphaActor and load state."""
    from algorithms.mappo import AdversaryAlphaActor
    actor = AdversaryAlphaActor(
        obs_dim=obs_dim, action_dim=action_dim,
        alpha_init=0.5, alpha_learnable=True,
    )
    try:
        actor.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"  load_state_dict warning: {e}")
    actor.eval()
    return actor


def collect_alphas(actor, env, n_episodes, seed):
    """Run episodes; record alpha output at every adv step.
    Returns: list[float] of all alpha values, list[list[float]] per-episode."""
    all_alphas = []
    per_ep = []

    rng = np.random.RandomState(seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.randint(0, 1_000_000)))
        ep_alphas = []

        for step in range(env.config.max_cycles):
            actions = {}
            for name in env.possible_agents:
                if not is_normal_adversary(name):
                    # For non-adv agents, take random action
                    spec = env.get_agent_spec(name)
                    actions[name] = int(rng.randint(spec.action_dim))
                    continue

                ob = obs[name]
                obs_t = torch.tensor(ob, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    own_logits, alpha = actor._compute_components(obs_t)
                    alpha_val = float(alpha.item())
                    ep_alphas.append(alpha_val)
                    all_alphas.append(alpha_val)

                    # take action via own_logits + plan_bias mix
                    cd = env.config.comm_dim
                    msg_onehot = ob[34:34 + cd]
                    if msg_onehot.sum() < 0.5:
                        message = 0
                    else:
                        message = int(np.argmax(msg_onehot))
                    plan_bias = env.compute_plan_bias(message, ob)
                    plan_bias_t = torch.tensor(plan_bias, dtype=torch.float32).unsqueeze(0)
                    final_logits = alpha * own_logits + (1.0 - alpha) * plan_bias_t
                    a = int(final_logits.argmax(dim=-1).item())
                    actions[name] = a

            obs, rewards, terms, truncs, infos = env.step(actions)
            if any(terms.values()) or any(truncs.values()):
                break

        per_ep.append(ep_alphas)

    return all_alphas, per_ep


def main():
    args = parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"ERROR: run_dir does not exist: {args.run_dir}")
        return

    print(f"Loading checkpoints from: {args.run_dir}")
    ckpts = find_checkpoints(args.run_dir)
    if not ckpts:
        print("No checkpoints found.")
        return
    print(f"Found {len(ckpts)} checkpoints")

    # Build env
    cfg = Config()
    cfg.condition = "E1_full"
    cfg.seed = args.seed
    env = make_env(cfg)
    env.reset(seed=args.seed)

    # Get adv obs/action dim from first agent
    adv_name = next(a for a in env.possible_agents if is_normal_adversary(a))
    spec = env.get_agent_spec(adv_name)
    obs_dim = spec.obs_dim
    action_dim = spec.action_dim
    print(f"Adv obs_dim={obs_dim} action_dim={action_dim}")

    # Collect alphas per checkpoint
    ep_nums = []
    means = []
    stds = []
    all_distributions = []  # list per ckpt
    final_per_ep_traces = None

    for i, (ep_num, folder) in enumerate(ckpts):
        print(f"\n[{i+1}/{len(ckpts)}] {os.path.basename(folder)} ...")
        state = load_actor_state(folder, role_name="adversary")
        if state is None:
            print("  ! no adversary actor file found, skipping")
            continue
        actor = build_actor_from_state(state, obs_dim, action_dim)

        all_alphas, per_ep = collect_alphas(actor, env, args.n_episodes, seed=args.seed + ep_num)
        if not all_alphas:
            print("  ! no alphas collected, skipping")
            continue

        arr = np.array(all_alphas)
        mean = float(arr.mean())
        std = float(arr.std())
        print(f"  alpha mean={mean:.3f} std={std:.3f} min={arr.min():.3f} max={arr.max():.3f} n={len(arr)}")

        ep_nums.append(ep_num)
        means.append(mean)
        stds.append(std)
        all_distributions.append(arr)

        if i == len(ckpts) - 1:
            final_per_ep_traces = per_ep

    # ─── Plot 1: mean alpha over training ─────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.plot(ep_nums, means_arr, marker='o', linewidth=2.5, color='#9467bd',
            label='Mean α across episodes & steps')
    ax.fill_between(ep_nums,
                     means_arr - stds_arr,
                     means_arr + stds_arr,
                     alpha=0.2, color='#9467bd', label='±1 std')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='α = 0.5 (initialization)')
    ax.set_title(r'Real $\alpha$ distribution over training (averaged over '
                 f'{args.n_episodes} eval episodes per checkpoint)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Training Episodes', fontsize=12)
    ax.set_ylabel(r'$\alpha$ value', fontsize=12)
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    fig.tight_layout()
    out1 = f"{args.out_prefix}_mean_trajectory.png"
    fig.savefig(out1, dpi=200, bbox_inches='tight')
    print(f"\n[saved] {out1}")
    plt.close(fig)

    # ─── Plot 2: distribution boxplot per checkpoint ──────
    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(all_distributions, positions=ep_nums,
                    widths=max(50, (max(ep_nums) - min(ep_nums)) / len(ep_nums) * 0.6),
                    showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#9467bd')
        patch.set_alpha(0.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(r'$\alpha$ distribution per checkpoint',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Training Episodes', fontsize=12)
    ax.set_ylabel(r'$\alpha$', fontsize=12)
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    out2 = f"{args.out_prefix}_distribution.png"
    fig.savefig(out2, dpi=200, bbox_inches='tight')
    print(f"[saved] {out2}")
    plt.close(fig)

    # ─── Plot 3: within-episode trace at final ckpt ───────
    if final_per_ep_traces:
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, trace in enumerate(final_per_ep_traces[:10]):
            ax.plot(trace, alpha=0.4, color='#9467bd')
        # mean trace (truncate to shortest)
        min_len = min(len(t) for t in final_per_ep_traces if len(t) > 0)
        if min_len > 0:
            stacked = np.array([t[:min_len] for t in final_per_ep_traces if len(t) >= min_len])
            mean_trace = stacked.mean(axis=0)
            ax.plot(mean_trace, color='#d62728', linewidth=2.5, label='Mean across episodes')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(r'Within-episode $\alpha$ trace at final checkpoint',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Step within episode', fontsize=12)
        ax.set_ylabel(r'$\alpha$', fontsize=12)
        ax.set_ylim([0.0, 1.0])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        fig.tight_layout()
        out3 = f"{args.out_prefix}_within_episode.png"
        fig.savefig(out3, dpi=200, bbox_inches='tight')
        print(f"[saved] {out3}")
        plt.close(fig)

    # ─── Save raw data ────────────────────────────────────
    np.savez(f"{args.out_prefix}_raw.npz",
             ep_nums=ep_nums, means=means, stds=stds,
             all_distributions=np.array(all_distributions, dtype=object),
             allow_pickle=True)
    print(f"[saved] {args.out_prefix}_raw.npz")

    print("\n" + "=" * 50)
    print("Summary across checkpoints:")
    print(f"  mean alpha range: {min(means):.3f} → {max(means):.3f}")
    print(f"  std alpha range:  {min(stds):.3f} → {max(stds):.3f}")
    print("=" * 50)
    print()
    print("Interpretation guide:")
    print("  • If mean ~ 0.5 and std small → alpha didn't really learn")
    print("  • If mean drifts to 0.7-0.9 → adv learned to TRUST own obs more")
    print("  • If mean drifts to 0.1-0.3 → adv learned to TRUST leader more")
    print("  • If std is large (>0.15) → alpha is state-dependent (good!)")


if __name__ == "__main__":
    main()
