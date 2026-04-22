"""
Robust Universal ExperimentRunner for MARL (PPO / MAPPO / DDPG / DQN / PSO-safe)
Fixes:
- NaN-safe logging
- metric isolation
- algorithm-agnostic update handling
- stable evaluation separation
"""

import os
import csv
import time
import numpy as np
from collections import defaultdict


class ExperimentRunner:

    def __init__(self, config, env, agent):
        self.config = config
        self.env = env
        self.agent = agent

        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.model_dir, exist_ok=True)

        log_path = os.path.join(config.log_dir, f"{config.exp_name}.csv")
        self._csv_file = open(log_path, "w", newline="")
        self._csv_writer = None

        self._global_step = 0
        self._start_time = time.time()

        # ===== safe metric buffer =====
        self._metrics = defaultdict(list)

    # =========================================================
    # MAIN LOOP
    # =========================================================
    def run(self):
        cfg = self.config

        print(f"\n[Experiment Start] {cfg.exp_name}")
        print(f"Episodes={cfg.total_episodes} | Update={cfg.update_every_n_episodes}")

        for ep in range(1, cfg.total_episodes + 1):

            ep_stats = self._run_episode(explore=True)
            self._global_step += ep_stats["steps"]

            self._accumulate(ep_stats)

            # =========================
            # UPDATE (ALGO-AGNOSTIC)
            # =========================
            if ep % cfg.update_every_n_episodes == 0:

                update_out = self.agent.update(self._global_step)

                # 🔥 SAFE UPDATE HANDLING
                if isinstance(update_out, dict):
                    self._accumulate(update_out)

            # =========================
            # EVAL
            # =========================
            if ep % cfg.eval_every_n_episodes == 0:
                eval_stats = self._run_eval()
                self._log_eval(ep, eval_stats)

            # =========================
            # TRAIN LOG
            # =========================
            if ep % cfg.print_every == 0:
                self._log_train(ep)
                self._metrics.clear()

            # =========================
            # SAVE
            # =========================
            if ep % cfg.save_every_n_episodes == 0:
                path = os.path.join(cfg.model_dir, cfg.exp_name, f"ep{ep:06d}")
                self.agent.save(path)
                print(f"[SAVE] {path}")

        self._csv_file.close()
        print("\n[Training Finished]")

    # =========================================================
    # EPISODE ROLLOUT
    # =========================================================
    def _run_episode(self, explore=True):

        obs, _ = self.env.reset()
        ep_reward = defaultdict(float)
        steps = 0

        while self.env.agents:

            actions = self.agent.select_actions(obs, self.env, explore=explore)

            actions = {a: actions[a] for a in self.env.agents if a in actions}

            next_obs, rewards, terms, truncs, infos = self.env.step(actions)

            # =========================
            # store transition
            # =========================
            if explore:
                self.agent.observe({
                    "obs": obs,
                    "actions": actions,
                    "rewards": rewards,
                    "next_obs": next_obs,
                    "terminated": terms,
                    "truncated": truncs,
                })

            # =========================
            # reward aggregation
            # =========================
            for k, v in rewards.items():
                ep_reward[k] += float(v)

            obs = next_obs
            steps += 1

        if explore:
            self.agent.end_episode()

        # =========================
        # SAFE ROLE SPLIT
        # =========================
        adv = getattr(self.agent, "adv_agents", [])
        good = getattr(self.agent, "good_agents", [])

        adv_r = np.mean([ep_reward[a] for a in adv if a in ep_reward]) if adv else 0.0
        good_r = np.mean([ep_reward[a] for a in good if a in ep_reward]) if good else 0.0

        return {
            "steps": steps,
            "adv_reward": float(adv_r),
            "good_reward": float(good_r),
        }

    # =========================================================
    # EVAL
    # =========================================================
    def _run_eval(self):
        cfg = self.config

        adv_r, good_r = [], []

        for _ in range(cfg.eval_episodes):
            stats = self._run_episode(explore=False)
            adv_r.append(stats["adv_reward"])
            good_r.append(stats["good_reward"])

        return {
            "eval_adv_mean": float(np.mean(adv_r)),
            "eval_adv_std": float(np.std(adv_r)),
            "eval_good_mean": float(np.mean(good_r)),
            "eval_good_std": float(np.std(good_r)),
        }

    # =========================================================
    # METRICS HANDLING (NaN SAFE CORE)
    # =========================================================
    def _accumulate(self, d):
        for k, v in d.items():
            if v is None:
                continue
            v = float(v)

            # 🔥 NaN / Inf filter (VERY IMPORTANT)
            if np.isnan(v) or np.isinf(v):
                continue

            self._metrics[k].append(v)

    def _mean(self, key):
        vals = self._metrics.get(key, [])
        return float(np.mean(vals)) if len(vals) > 0 else None

    def _safe(self, v):
        if v is None or np.isnan(v) or np.isinf(v):
            return "nan"
        return f"{v:.4f}"

    # =========================================================
    # TRAIN LOG
    # =========================================================
    def _log_train(self, ep):

        elapsed = time.time() - self._start_time

        row = {
            "episode": ep,
            "step": self._global_step,
            "time_s": f"{elapsed:.1f}",

            "adv_reward": self._safe(self._mean("adv_reward")),
            "good_reward": self._safe(self._mean("good_reward")),

            # losses (multi-algo safe)
            "actor_loss": self._safe(self._mean("actor_loss")),
            "critic_loss": self._safe(self._mean("critic_loss")),
            "q_loss": self._safe(self._mean("q_loss")),

            "entropy": self._safe(self._mean("entropy")),
        }

        print(
            f"[EP {ep:5d}] "
            f"adv={row['adv_reward']} good={row['good_reward']} "
            f"actor={row['actor_loss']} critic={row['critic_loss']} "
            f"q={row['q_loss']} time={row['time_s']}s"
        )

        self._write(row)

    # =========================================================
    # EVAL LOG
    # =========================================================
    def _log_eval(self, ep, stats):
        print(
            f"[EVAL {ep}] "
            f"adv={stats['eval_adv_mean']:.3f}±{stats['eval_adv_std']:.3f} "
            f"good={stats['eval_good_mean']:.3f}±{stats['eval_good_std']:.3f}"
        )

        self._write({"episode": ep, "type": "eval", **stats})

    # =========================================================
    # CSV
    # =========================================================
    def _write(self, row):

        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=list(row.keys()),
                extrasaction="ignore"
            )
            self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()