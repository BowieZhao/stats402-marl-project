"""
Experiment Runner for the FINAL design.

Loop:
    reset env → for each step: select actions, env.step, agent.observe
    Every config.update_every episodes: agent.update()
    Every config.eval_every episodes: run a deterministic eval rollout
    Save checkpoints periodically.
"""

import os
import csv
import time
import json
import numpy as np
from collections import defaultdict

from envs import is_good, is_leader, is_normal_adversary
from frozen_policy import FrozenGoodPolicy


class Runner:
    def __init__(self, config, env, agent):
        self.config = config
        self.env = env
        self.agent = agent

        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.model_dir, exist_ok=True)

        # Optional frozen good policy
        self.frozen_good = None
        if config.frozen_good_path is not None:
            good_names = [a for a in env.possible_agents if is_good(a)]
            if good_names:
                spec = env.get_agent_spec(good_names[0])
                self.frozen_good = FrozenGoodPolicy(
                    config.frozen_good_path,
                    obs_dim=spec.obs_dim,
                    action_dim=spec.action_dim,
                    deterministic=config.frozen_good_deterministic,
                    continuous=config.continuous_actions,
                )
                print(f"[Runner] Frozen good policy ACTIVE.")
                print(f"[Runner] Only adversary-side will learn.")

        # CSV logging
        log_path = os.path.join(config.log_dir, f"{config.exp_name}.csv")
        self._csv_file = open(log_path, "w", newline="")
        self._csv_writer = None

        # Save config snapshot
        cfg_path = os.path.join(config.log_dir, f"{config.exp_name}_config.json")
        with open(cfg_path, "w") as f:
            cfg_dict = {k: v for k, v in vars(config).items()
                        if not k.startswith("_") and not callable(v)}
            json.dump(cfg_dict, f, indent=2, default=str)

        self._global_step = 0
        self._start_time = time.time()
        self._metrics = defaultdict(list)

        self._good_names = [a for a in env.possible_agents if is_good(a)]
        self._adv_names = [a for a in env.possible_agents if not is_good(a)]

    def run(self):
        cfg = self.config
        print(f"\n[Experiment Start] {cfg.exp_name}")
        print(f"Episodes={cfg.total_episodes} | Update every {cfg.update_every}")

        for ep in range(1, cfg.total_episodes + 1):
            ep_stats = self._run_episode(explore=True)
            self._global_step += ep_stats["steps"]
            self._accumulate(ep_stats)

            if ep % cfg.update_every == 0:
                update_out = self.agent.update(self._global_step)
                if isinstance(update_out, dict):
                    self._accumulate(update_out)

            if ep % cfg.eval_every == 0:
                eval_stats = self._run_eval()
                self._log_eval(ep, eval_stats)

            if ep % cfg.log_every == 0:
                self._log_train(ep)
                self._metrics.clear()

            if ep % cfg.save_every == 0:
                path = os.path.join(cfg.model_dir, cfg.exp_name, f"ep{ep:06d}")
                self.agent.save(path)
                print(f"[SAVE] {path}")

        final_path = os.path.join(cfg.model_dir, cfg.exp_name, "final")
        self.agent.save(final_path)
        print(f"[SAVE FINAL] {final_path}")

        self._csv_file.close()
        print("\n[Training Finished]")

    def _get_actions(self, obs, explore: bool):
        if self.frozen_good is not None:
            adv_obs = {a: obs[a] for a in self._adv_names if a in obs}
            adv_actions = self.agent.select_actions(adv_obs, self.env, explore=explore)

            good_actions = {}
            for g in self._good_names:
                if g in obs:
                    good_actions[g] = self.frozen_good.act(obs[g])
            return {**adv_actions, **good_actions}
        else:
            return self.agent.select_actions(obs, self.env, explore=explore)

    def _run_episode(self, explore=True):
        env = self.env
        obs, _ = env.reset()

        ep_returns = defaultdict(float)
        steps = 0
        encircle_steps = 0   # number of steps with ≥3 advs around any uncaught good

        for _ in range(self.config.max_cycles):
            actions = self._get_actions(obs, explore=explore)
            next_obs, rewards, terms, truncs, infos = env.step(actions)

            transition = {
                "obs": obs,
                "actions": actions,
                "rewards": rewards,
                "next_obs": next_obs,
                "terminated": terms,
                "truncated": truncs,
            }

            # Filter out frozen-good transitions before storage
            if self.frozen_good is not None:
                transition_for_agent = {
                    "obs": {a: o for a, o in obs.items() if not is_good(a)},
                    "actions": {a: ac for a, ac in actions.items() if not is_good(a)},
                    "rewards": {a: r for a, r in rewards.items() if not is_good(a)},
                    "next_obs": {a: o for a, o in next_obs.items() if not is_good(a)},
                    "terminated": {a: t for a, t in terms.items() if not is_good(a)},
                    "truncated": {a: t for a, t in truncs.items() if not is_good(a)},
                }
                self.agent.observe(transition_for_agent)
            else:
                self.agent.observe(transition)

            for a, r in rewards.items():
                ep_returns[a] += float(r)

            # Track encircle (≥3 advs around any uncaught good)
            try:
                positions = env._get_agent_positions()
                adv_names = env._leader_names + env._normal_adv_names
                uncaught = [g for g in env._good_names
                            if g not in env._goods_caught_ever]
                for g in uncaught:
                    if g not in positions: continue
                    n_in = sum(1 for a in adv_names
                               if a in positions and
                               float(np.linalg.norm(positions[a] - positions[g])) < env.config.R_encircle_radius)
                    if n_in >= env.config.R_encircle_min_count:
                        encircle_steps += 1
                        break  # one encircle per step is enough
            except Exception:
                pass

            obs = next_obs
            steps += 1

            if any(terms.values()) or any(truncs.values()):
                break

        self.agent.end_episode()

        adv_return = np.mean([ep_returns[a] for a in self._adv_names
                               if a in ep_returns]) if self._adv_names else 0.0
        good_return = np.mean([ep_returns[a] for a in self._good_names
                                if a in ep_returns]) if self._good_names else 0.0
        adv_total = sum(ep_returns[a] for a in self._adv_names if a in ep_returns)

        # Catches this episode: how many goods got first-caught
        catches = len(env._goods_caught_ever) if hasattr(env, "_goods_caught_ever") else 0

        return {
            "adv_return": adv_return,
            "adv_total": adv_total,
            "good_return": good_return,
            "steps": steps,
            "catches": catches,
            "encircle_steps": encircle_steps,
        }

    def _run_eval(self):
        n = self.config.eval_episodes
        adv_returns, good_returns = [], []
        adv_totals, catches_list, encircle_list = [], [], []

        # Switch to eval mode: skip teaching rewards (role_align, thrash, correct_msg)
        self.env.set_eval_mode(True)
        try:
            for _ in range(n):
                stats = self._run_episode(explore=False)
                adv_returns.append(stats["adv_return"])
                good_returns.append(stats["good_return"])
                adv_totals.append(stats["adv_total"])
                catches_list.append(stats["catches"])
                encircle_list.append(stats["encircle_steps"])
        finally:
            self.env.set_eval_mode(False)  # always restore
        return {
            "eval_adv_mean": float(np.mean(adv_returns)),
            "eval_adv_std": float(np.std(adv_returns)),
            "eval_adv_total_mean": float(np.mean(adv_totals)),
            "eval_good_mean": float(np.mean(good_returns)),
            "eval_good_std": float(np.std(good_returns)),
            "eval_catches_mean": float(np.mean(catches_list)),
            "eval_encircle_mean": float(np.mean(encircle_list)),
        }

    def _accumulate(self, d):
        for k, v in d.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                self._metrics[k].append(float(v))

    def _log_train(self, ep):
        elapsed = time.time() - self._start_time
        adv = float(np.mean(self._metrics.get("adv_return", [0])))
        good = float(np.mean(self._metrics.get("good_return", [0])))
        actor = float(np.mean(self._metrics.get("adversary/actor_loss", [float("nan")])))
        critic = float(np.mean(self._metrics.get("adversary/critic_loss", [float("nan")])))
        ent = float(np.mean(self._metrics.get("adversary/entropy", [float("nan")])))

        print(f"[EP {ep:5d}] adv={adv:.4f} good={good:.4f} "
              f"actor={actor:.4f} critic={critic:.4f} ent={ent:.4f} "
              f"time={elapsed:.1f}s")

        # Write CSV
        row = {
            "episode": ep,
            "adv_return": adv,
            "good_return": good,
            "actor_loss": actor,
            "critic_loss": critic,
            "entropy": ent,
            "elapsed": elapsed,
        }
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _log_eval(self, ep, eval_stats):
        adv_m = eval_stats["eval_adv_mean"]
        adv_s = eval_stats["eval_adv_std"]
        adv_total = eval_stats["eval_adv_total_mean"]
        good_m = eval_stats["eval_good_mean"]
        catches = eval_stats["eval_catches_mean"]
        encircle = eval_stats["eval_encircle_mean"]
        print(f"[EVAL {ep}] team_total={adv_total:7.1f}  "
              f"adv_avg={adv_m:6.2f}±{adv_s:5.2f}  "
              f"catches={catches:.2f}/2  "
              f"encircle_steps={encircle:.1f}/{self.config.max_cycles}  "
              f"good={good_m:6.2f}")
