"""
Visualize tool for inspecting trained MAPPO behavior.

Usage:
    python visualize.py --checkpoint outputs/models/sanity_mappo_E1_full_s42/final
    python visualize.py --checkpoint <path> --gif       # save GIF
    python visualize.py --checkpoint <path> --breakdown # print per-step reward breakdown

Output:
    - Console: per-step reward decomposition (R_distance / R_capture / R_encircle / R_progress / R_role_align)
    - Console: encircle stats (% of steps with ≥3 advs around a good)
    - Optional: rollout.gif (rendered episode)
"""

import argparse
import os
import sys
import numpy as np

from config import Config
from envs import make_env, is_good, is_normal_adversary
from algorithms.mappo import MAPPOAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint dir (e.g. outputs/models/.../final)")
    p.add_argument("--condition", default="E1_full",
                   choices=["E1_full", "E2_no_comm", "E3_no_alpha"])
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--max_cycles", type=int, default=50)
    p.add_argument("--num_good", type=int, default=2)
    p.add_argument("--num_forests", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gif", action="store_true",
                   help="Save rendered episode as GIF")
    p.add_argument("--breakdown", action="store_true",
                   help="Print per-step reward breakdown")
    p.add_argument("--frozen_good", default=None)
    return p.parse_args()


def reward_breakdown(env, action_dict, leader_message, new_first_catches):
    """Recompute reward components separately for inspection."""
    cfg = env.config
    positions = env._get_agent_positions()
    adv_names = env._leader_names + env._normal_adv_names
    good_names = env._good_names

    components = {a: {"distance": 0, "capture": 0, "encircle": 0,
                       "progress": 0, "pin_assist": 0, "team_bonus": 0,
                       "role_align": 0, "thrash": 0}
                  for a in adv_names}

    # R_distance
    for a in adv_names:
        if a not in positions: continue
        min_d = min(np.linalg.norm(positions[a] - positions[g])
                    for g in good_names if g in positions)
        components[a]["distance"] = cfg.R_distance_coef * float(min_d)

    # R_capture
    for a in adv_names:
        if a not in positions: continue
        for g in good_names:
            if g not in positions: continue
            d = float(np.linalg.norm(positions[a] - positions[g]))
            if d < cfg.collision_distance:
                components[a]["capture"] = cfg.R_capture
                break

    # R_encircle
    for g in good_names:
        if g not in positions: continue
        in_group = []
        for a in adv_names:
            if a not in positions: continue
            d = float(np.linalg.norm(positions[a] - positions[g]))
            if d < cfg.R_encircle_radius:
                in_group.append(a)
        n = len(in_group)
        if n >= cfg.R_encircle_min_count:
            bonus = cfg.R_encircle_per_extra * (n - 1)
            for a in in_group:
                components[a]["encircle"] += bonus

    # R_progress + R_pin_assist + R_team_bonus on first catches
    if len(new_first_catches) > 0:
        # Determine 1st vs 2nd catch
        # Note: at this point env._goods_caught_ever already includes the new catches
        # So n_already counts catches BEFORE this step
        n_caught_total = len(env._goods_caught_ever)
        for i, caught_g in enumerate(new_first_catches):
            if caught_g not in positions: continue
            g_pos = positions[caught_g]

            # Find killer
            killer = None
            for a in adv_names:
                if a not in positions: continue
                d = float(np.linalg.norm(positions[a] - g_pos))
                if d < cfg.collision_distance:
                    killer = a
                    break

            # Determine catch rank (1st or 2nd)
            n_already = n_caught_total - len(new_first_catches) + i
            if n_already == 0:
                progress_reward = cfg.R_progress_first
            else:
                progress_reward = cfg.R_progress_second

            # Award progress to killer
            if killer is not None:
                components[killer]["progress"] += progress_reward

            # Pin_assist to encircle group (not killer)
            for a in adv_names:
                if a not in positions or a == killer: continue
                d = float(np.linalg.norm(positions[a] - g_pos))
                if d < cfg.R_encircle_radius:
                    components[a]["pin_assist"] = components[a].get("pin_assist", 0) + cfg.R_pin_assist

            # Team bonus to all
            for a in adv_names:
                components[a]["team_bonus"] = components[a].get("team_bonus", 0) + cfg.R_team_bonus

    # R_role_align
    if cfg.plan_bias_enabled:
        for a in env._normal_adv_names:
            if a not in action_dict: continue
            action = int(action_dict[a])
            if a in positions:
                base_obs = env._reconstruct_adv_base_obs(a, positions)
                bias = env.compute_plan_bias(leader_message, base_obs)
                if np.max(bias) > 0 and action == int(np.argmax(bias)):
                    components[a]["role_align"] = cfg.R_role_align

    return components


def encircle_stats(env, positions):
    """Returns dict: { good_name: n_advs_within_radius }"""
    cfg = env.config
    adv_names = env._leader_names + env._normal_adv_names
    good_names = env._good_names

    out = {}
    for g in good_names:
        if g not in positions:
            out[g] = 0
            continue
        n = 0
        for a in adv_names:
            if a not in positions: continue
            d = float(np.linalg.norm(positions[a] - positions[g]))
            if d < cfg.R_encircle_radius:
                n += 1
        out[g] = n
    return out


def main():
    args = parse_args()

    cfg = Config()
    cfg.condition = args.condition
    cfg.max_cycles = args.max_cycles
    cfg.num_good = args.num_good
    cfg.num_forests = args.num_forests
    cfg.seed = args.seed
    cfg.frozen_good_path = args.frozen_good
    if args.gif:
        cfg.render_mode = "rgb_array"
    cfg.exp_name = "viz"

    env = make_env(cfg)

    print(f"\n[Viz] Loading checkpoint from: {args.checkpoint}")
    agent = MAPPOAgent(cfg, env)
    agent.load(args.checkpoint)
    print("[Viz] Checkpoint loaded.\n")

    # Frames for GIF
    frames = []

    # Stats accumulators across episodes
    all_returns = []
    all_breakdown = {"distance": 0, "capture": 0, "encircle": 0,
                      "progress": 0, "pin_assist": 0, "team_bonus": 0,
                      "role_align": 0, "thrash": 0}
    encircle_counter = {2: 0, 3: 0, 4: 0}  # how many steps had 2/3/4 advs around any good
    total_steps = 0
    catches = 0

    for ep in range(1, args.n_episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_return = {a: 0.0 for a in env.possible_agents}
        prev_caught_ever = set()

        print(f"\n--- Episode {ep} ---")

        for step in range(args.max_cycles):
            actions = agent.select_actions(obs, env, explore=False)
            leader_msg = env._extract_leader_message(actions)

            # Step env (this updates env._caught_set_this_step internally)
            next_obs, rewards, terms, truncs, _ = env.step(actions)

            # New first-catches this step
            new_catches = env._goods_caught_ever - prev_caught_ever
            prev_caught_ever = set(env._goods_caught_ever)

            # Render
            if args.gif:
                try:
                    img = env._env.render()
                    if img is not None:
                        frames.append(img)
                except Exception as e:
                    print(f"render failed: {e}")

            # Breakdown
            if args.breakdown:
                comp = reward_breakdown(env, actions, leader_msg, new_catches)
                # Aggregate adv-side
                step_total = {k: 0 for k in all_breakdown}
                for a, c in comp.items():
                    if "leadadversary" in a or is_normal_adversary(a):
                        for k, v in c.items():
                            step_total[k] += v

                msg_str = ["A", "B", "C", "D"][leader_msg]
                positions = env._get_agent_positions()
                ec = encircle_stats(env, positions)

                if any(step_total.values()) or new_catches:
                    print(f"  step{step:2d}: msg={msg_str} "
                          f"dist={step_total['distance']:+5.2f} "
                          f"cap={step_total['capture']:+4.0f} "
                          f"enc={step_total['encircle']:+4.0f} "
                          f"prog={step_total['progress']:+4.0f} "
                          f"pin={step_total['pin_assist']:+4.0f} "
                          f"team={step_total['team_bonus']:+4.0f} "
                          f"role={step_total['role_align']:+4.1f}"
                          f"  | g0_n={ec.get('agent_0', 0)} g1_n={ec.get('agent_1', 0)}"
                          f"  | new_catches={list(new_catches)}")

            # Encircle counter
            positions = env._get_agent_positions()
            ec_count = encircle_stats(env, positions)
            for g, n in ec_count.items():
                if n >= 2: encircle_counter[2] += 1
                if n >= 3: encircle_counter[3] += 1
                if n >= 4: encircle_counter[4] += 1

            # Reward accumulation
            for a, r in rewards.items():
                ep_return[a] += float(r)

            # Aggregate breakdown across episode
            if args.breakdown:
                for k, v in step_total.items():
                    all_breakdown[k] += v

            obs = next_obs
            total_steps += 1
            catches += len(new_catches)

            if any(terms.values()) or any(truncs.values()):
                break

        adv_total = np.mean([ep_return[a] for a in env.possible_agents
                              if not is_good(a) and a in ep_return])
        good_total = np.mean([ep_return[a] for a in env.possible_agents
                               if is_good(a) and a in ep_return])
        print(f"  Episode return: adv={adv_total:.2f}  good={good_total:.2f}  "
              f"catches={len(prev_caught_ever)}")
        all_returns.append(adv_total)

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Episodes:      {args.n_episodes}")
    print(f"Avg adv return: {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
    print(f"Total goods caught: {catches} / {args.n_episodes * cfg.num_good} possible")
    print(f"Catch rate:    {100*catches/(args.n_episodes * cfg.num_good):.0f}%")

    print(f"\nEncircle frequency (out of {total_steps} steps × 2 goods = "
          f"{2*total_steps} (good, step) pairs):")
    print(f"  ≥2 advs at 0.4 radius:  {encircle_counter[2]:5d}  "
          f"({100*encircle_counter[2]/(2*total_steps):.1f}%)")
    print(f"  ≥3 advs (R_encircle):   {encircle_counter[3]:5d}  "
          f"({100*encircle_counter[3]/(2*total_steps):.1f}%)")
    print(f"  ≥4 advs (full encircle):{encircle_counter[4]:5d}  "
          f"({100*encircle_counter[4]/(2*total_steps):.1f}%)")

    if args.breakdown:
        print(f"\nReward breakdown (cumulative across {args.n_episodes} eps, "
              f"adv-side total):")
        total_r = sum(all_breakdown.values())
        for k, v in all_breakdown.items():
            pct = 100 * v / total_r if total_r else 0
            print(f"  {k:12s} = {v:+10.1f}  ({pct:+5.1f}%)")
        print(f"  {'TOTAL':12s} = {total_r:+10.1f}")

    # ── Save GIF ──
    if args.gif and frames:
        try:
            from PIL import Image
            print(f"\nSaving GIF with {len(frames)} frames...")
            imgs = [Image.fromarray(f) for f in frames]
            out_path = "rollout.gif"
            imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                          duration=80, loop=0)
            print(f"GIF saved to: {out_path}")
        except ImportError:
            print("PIL not installed: pip install pillow")

    env.close()


if __name__ == "__main__":
    main()
