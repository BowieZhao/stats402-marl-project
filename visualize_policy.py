"""
Visualize trained MAPPO E1 policy.

Loads the final checkpoint and runs episodes with rendering.
Optionally saves a GIF for use in poster / paper.

Usage:
    python visualize_policy.py                    # render to screen
    python visualize_policy.py --save_gif         # also save a GIF
    python visualize_policy.py --n_episodes 5     # render N episodes
"""

import os
import argparse
import numpy as np
import torch

from config import Config
from envs import make_env, is_normal_adversary, is_leader, is_good


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=None,
                   help="Checkpoint dir. Defaults to NEW_PPO_KING/final.")
    p.add_argument("--n_episodes", type=int, default=3)
    p.add_argument("--save_gif", action="store_true",
                   help="Save GIF to ./episode_X.gif")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_cycles", type=int, default=50)
    p.add_argument("--no_render", action="store_true",
                   help="Don't open render window (only useful with --save_gif)")
    return p.parse_args()


def load_role_actor(ckpt_dir, role_name, obs_dim, action_dim,
                     alpha_init=0.5, alpha_learnable=True):
    """Load actor for a role from checkpoint."""
    candidates = [
        os.path.join(ckpt_dir, f"{role_name}.pt"),
        os.path.join(ckpt_dir, f"{role_name}_actor.pt"),
    ]
    state = None
    for c in candidates:
        if os.path.exists(c):
            try:
                data = torch.load(c, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(c, map_location="cpu")
            if isinstance(data, dict):
                if "model" in data: state = data["model"]
                elif "actor" in data: state = data["actor"]
                elif "state_dict" in data: state = data["state_dict"]
                else: state = data
            else:
                state = data
            break

    if state is None:
        print(f"  ! no actor file found for role={role_name} in {ckpt_dir}")
        return None

    if role_name == "adversary":
        from algorithms.mappo import AdversaryAlphaActor
        actor = AdversaryAlphaActor(
            obs_dim=obs_dim, action_dim=action_dim,
            alpha_init=alpha_init, alpha_learnable=alpha_learnable,
        )
    else:
        from algorithms.mappo import ActorNet
        actor = ActorNet(obs_dim=obs_dim, action_dim=action_dim)

    try:
        actor.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"  load warning: {e}")
    actor.eval()
    return actor


def select_action(role, name, obs, actors, env):
    """Pick a deterministic action for one agent."""
    actor = actors.get(role)
    if actor is None:
        return 0  # fallback noop

    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        if role == "adversary":
            own_logits, alpha = actor._compute_components(obs_t)
            cd = env.config.comm_dim
            msg_onehot = obs[34:34 + cd]
            if msg_onehot.sum() < 0.5:
                message = 0
            else:
                message = int(np.argmax(msg_onehot))
            plan_bias = env.compute_plan_bias(message, obs)
            plan_bias_t = torch.tensor(plan_bias, dtype=torch.float32).unsqueeze(0)
            final_logits = alpha * own_logits + (1.0 - alpha) * plan_bias_t
            return int(final_logits.argmax(dim=-1).item()), float(alpha.item()), message
        else:
            logits = actor(obs_t)
            return int(logits.argmax(dim=-1).item()), None, None


def main():
    args = parse_args()

    cfg = Config()
    cfg.condition = "E1_full"
    cfg.seed = args.seed
    cfg.max_cycles = args.max_cycles
    cfg.render_mode = None if args.no_render else "rgb_array"  # always use rgb_array if we want frames

    # Build env
    env = make_env(cfg)
    obs, _ = env.reset(seed=args.seed)

    # Default ckpt
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(
            "outputs", "models",
            "NEW_PPO_KING_mappo_E1_full_s42", "final",
        )
    print(f"Loading checkpoint from: {args.ckpt_dir}")

    # Get specs
    leader_name = next(a for a in env.possible_agents if is_leader(a))
    adv_name    = next(a for a in env.possible_agents if is_normal_adversary(a))
    good_name   = next(a for a in env.possible_agents if is_good(a))

    leader_spec = env.get_agent_spec(leader_name)
    adv_spec    = env.get_agent_spec(adv_name)
    good_spec   = env.get_agent_spec(good_name)

    # Load actors per role
    print("Loading actors...")
    actors = {}
    a_leader = load_role_actor(args.ckpt_dir, "leader",
                                 leader_spec.obs_dim, leader_spec.action_dim)
    a_adv = load_role_actor(args.ckpt_dir, "adversary",
                              adv_spec.obs_dim, adv_spec.action_dim,
                              alpha_init=cfg.alpha_init,
                              alpha_learnable=cfg.alpha_is_learnable)
    a_good = load_role_actor(args.ckpt_dir, "good",
                               good_spec.obs_dim, good_spec.action_dim)
    if a_leader: actors["leader"] = a_leader
    if a_adv:    actors["adversary"] = a_adv
    if a_good:   actors["good"] = a_good

    if not actors:
        print("ERROR: no actors loaded. Check ckpt_dir.")
        return

    # Optional GIF saver
    if args.save_gif:
        try:
            from PIL import Image
            HAS_PIL = True
        except ImportError:
            print("WARN: pillow not installed, can't save GIF. Run: pip install pillow")
            HAS_PIL = False
    else:
        HAS_PIL = False

    rng = np.random.RandomState(args.seed)
    msg_names = ['A', 'B', 'C', 'D', 'E']

    for ep in range(args.n_episodes):
        ep_seed = int(rng.randint(0, 1_000_000))
        obs, _ = env.reset(seed=ep_seed)
        frames = []
        msg_history = []
        alpha_history = []
        catches = 0

        print(f"\n=== Episode {ep + 1} (seed={ep_seed}) ===")

        for step in range(cfg.max_cycles):
            actions = {}
            ep_alphas_this_step = []
            current_msg = None

            for name in env.possible_agents:
                if is_leader(name):
                    role = "leader"
                elif is_normal_adversary(name):
                    role = "adversary"
                else:
                    role = "good"

                if role == "adversary":
                    a, alpha_val, msg = select_action(role, name, obs[name], actors, env)
                    ep_alphas_this_step.append(alpha_val)
                    current_msg = msg
                else:
                    a, _, _ = select_action(role, name, obs[name], actors, env)
                actions[name] = a

            # Capture frame BEFORE step (so we see the action being decided)
            try:
                frame = env._env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                pass

            obs, rewards, terms, truncs, infos = env.step(actions)

            mean_alpha = float(np.mean(ep_alphas_this_step)) if ep_alphas_this_step else 0.0
            msg_str = msg_names[current_msg] if current_msg is not None and current_msg < 5 else "?"

            n_caught = len(env._goods_caught_ever) if hasattr(env, "_goods_caught_ever") else 0
            if n_caught > catches:
                catches = n_caught
                print(f"  step {step:2d}: msg={msg_str} alpha={mean_alpha:.2f} "
                      f"  ▲ CATCH! ({catches}/{cfg.num_good})")
            else:
                print(f"  step {step:2d}: msg={msg_str} alpha={mean_alpha:.2f}")

            msg_history.append(current_msg)
            alpha_history.append(mean_alpha)

            if any(terms.values()) or any(truncs.values()):
                break

        # Final frame
        try:
            frame = env._env.render()
            if frame is not None:
                frames.append(frame)
        except Exception:
            pass

        print(f"  → Episode end: {catches}/{cfg.num_good} goods caught")
        print(f"  → Messages: {[msg_names[m] if m is not None and m < 5 else '?' for m in msg_history]}")

        # Save GIF
        if HAS_PIL and frames:
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]
            gif_path = f"episode_{ep+1}_seed{ep_seed}_catches{catches}.gif"
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=150,
                loop=0,
            )
            print(f"  → saved GIF: {gif_path}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
