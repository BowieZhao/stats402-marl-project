"""
ExperimentRunner for MARL (MAPPO / DQN / DDPG / PSO).

Handles two modes:
- Standard: all agents (leader + adversaries + goods) learn together.
- Frozen-good: the good agents use a pre-loaded FrozenGoodPolicy; only the
  adversary-side is trained. This enables fair cross-algorithm comparison
  because every experiment faces an identical opponent.

Frozen-good specifics:
- good actions come from FrozenGoodPolicy, never from the learning agent.
- The learning agent's observe() is called only with adversary-side transitions
  (good rewards / transitions are filtered out).
"""

import os
import csv
import time
import numpy as np
from collections import defaultdict

from envs import is_good, is_leader, is_adversary


class ExperimentRunner:

    def __init__(self, config, env, agent, frozen_good=None):
        self.config = config
        self.env = env
        self.agent = agent
        self.frozen_good = frozen_good  # None or FrozenGoodPolicy instance

        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.model_dir, exist_ok=True)

        log_path = os.path.join(config.log_dir, f"{config.exp_name}.csv")
        self._csv_file = open(log_path, "w", newline="")
        self._csv_writer = None

        self._global_step = 0
        self._start_time = time.time()
        self._metrics = defaultdict(list)

        # Identify good vs adversary names once
        self._good_names = [a for a in env.possible_agents if is_good(a)]
        self._adv_names = [a for a in env.possible_agents if not is_good(a)]

        if frozen_good is not None:
            print(f"[Runner] Frozen good policy ACTIVE. "
                  f"Goods ({len(self._good_names)}) will use fixed policy.")
            print(f"[Runner] Only adversary-side ({len(self._adv_names)}) will learn.")

    # ──────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────

    def run(self):
        cfg = self.config
        print(f"\n[Experiment Start] {cfg.exp_name}")
        print(f"Episodes={cfg.total_episodes} | Update={cfg.update_every_n_episodes}")

        for ep in range(1, cfg.total_episodes + 1):

            ep_stats = self._run_episode(explore=True)
            self._global_step += ep_stats["steps"]
            self._accumulate(ep_stats)

            if ep % cfg.update_every_n_episodes == 0:
                update_out = self.agent.update(self._global_step)
                if isinstance(update_out, dict):
                    self._accumulate(update_out)

            if ep % cfg.eval_every_n_episodes == 0:
                eval_stats = self._run_eval()
                self._log_eval(ep, eval_stats)

            if ep % cfg.print_every == 0:
                self._log_train(ep)
                self._metrics.clear()

            if ep % cfg.save_every_n_episodes == 0:
                path = os.path.join(cfg.model_dir, cfg.exp_name, f"ep{ep:06d}")
                self.agent.save(path)
                print(f"[SAVE] {path}")

        final_path = os.path.join(cfg.model_dir, cfg.exp_name, "final")
        self.agent.save(final_path)
        print(f"[SAVE FINAL] {final_path}")

        self._csv_file.close()
        print("\n[Training Finished]")

    # ──────────────────────────────────────────────────────────
    # Episode rollout
    # ──────────────────────────────────────────────────────────

    def _get_actions(self, obs, explore):
        """
        Get actions for all alive agents.
        - If frozen_good is set: goods use frozen policy, others use learning agent.
        - Otherwise: all actions from learning agent.
        """
        if self.frozen_good is None:
            actions = self.agent.select_actions(obs, self.env, explore=explore)
            return {a: actions[a] for a in self.env.agents if a in actions}

        # Frozen-good mode: split
        actions = {}

        # Adversary-side from learning agent
        adv_obs = {a: o for a, o in obs.items() if not is_good(a)}
        if adv_obs:
            adv_actions = self.agent.select_actions(adv_obs, self.env, explore=explore)
            for a, act in adv_actions.items():
                if a in self.env.agents:
                    actions[a] = act

        # Good-side from frozen policy
        for g in self._good_names:
            if g in obs and g in self.env.agents:
                actions[g] = self.frozen_good.act(obs[g])

        return actions

    def _filter_for_agent(self, transition):
        """
        When frozen_good is set, strip goods out of the transition before
        passing to the learning agent. This prevents the learning agent from
        storing good transitions in its buffer or updating good policies.
        """
        if self.frozen_good is None:
            return transition

        filtered = {}
        for key in ("obs", "actions", "rewards", "next_obs"):
            src = transition.get(key, {})
            filtered[key] = {a: v for a, v in src.items() if not is_good(a)}
        for key in ("terminated", "truncated"):
            src = transition.get(key, {})
            filtered[key] = {a: v for a, v in src.items() if not is_good(a)}
        return filtered

    def _run_episode(self, explore=True):
        obs, _ = self.env.reset()
        ep_reward = defaultdict(float)
        steps = 0

        while self.env.agents:
            actions = self._get_actions(obs, explore)

            next_obs, rewards, terms, truncs, infos = self.env.step(actions)

            if explore:
                raw = {
                    "obs": obs,
                    "actions": actions,
                    "rewards": rewards,
                    "next_obs": next_obs,
                    "terminated": terms,
                    "truncated": truncs,
                }
                self.agent.observe(self._filter_for_agent(raw))

            for k, v in rewards.items():
                ep_reward[k] += float(v)

            obs = next_obs
            steps += 1

        if explore and hasattr(self.agent, "end_episode"):
            self.agent.end_episode()

        # Use env-level names, not agent.adv_agents, because in frozen-good mode
        # the learning agent may not even track good names.
        adv_r = (np.mean([ep_reward[a] for a in self._adv_names if a in ep_reward])
                 if self._adv_names else 0.0)
        good_r = (np.mean([ep_reward[a] for a in self._good_names if a in ep_reward])
                  if self._good_names else 0.0)

        return {
            "steps": steps,
            "adv_reward": float(adv_r),
            "good_reward": float(good_r),
        }

    # ──────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────
    # Metric handling (NaN-safe)
    # ──────────────────────────────────────────────────────────

    def _accumulate(self, d):
        for k, v in d.items():
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(v) or np.isinf(v):
                continue
            self._metrics[k].append(v)

    def _mean(self, key):
        vals = self._metrics.get(key, [])
        return float(np.mean(vals)) if len(vals) > 0 else None

    def _safe(self, v):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "nan"
        return f"{v:.4f}"

    # ──────────────────────────────────────────────────────────
    # Training log
    # ──────────────────────────────────────────────────────────

    def _log_train(self, ep):
        elapsed = time.time() - self._start_time

        def _avg_keys_ending(suffix):
            vals = []
            for k in self._metrics:
                if k.endswith(suffix):
                    m = self._mean(k)
                    if m is not None:
                        vals.append(m)
            return float(np.mean(vals)) if vals else None

        actor = _avg_keys_ending("/actor_loss") or self._mean("actor_loss")
        critic = _avg_keys_ending("/critic_loss") or self._mean("critic_loss")
        entropy = _avg_keys_ending("/entropy") or self._mean("entropy")
        kl = _avg_keys_ending("/approx_kl") or self._mean("approx_kl")
        q_loss = _avg_keys_ending("/q_loss") or self._mean("q_loss")

        row = {
            "episode": ep,
            "step": self._global_step,
            "time_s": f"{elapsed:.1f}",
            "adv_reward": self._safe(self._mean("adv_reward")),
            "good_reward": self._safe(self._mean("good_reward")),
            "actor_loss": self._safe(actor),
            "critic_loss": self._safe(critic),
            "entropy": self._safe(entropy),
            "kl": self._safe(kl),
            "q_loss": self._safe(q_loss),
        }

        print(
            f"[EP {ep:5d}] "
            f"adv={row['adv_reward']} good={row['good_reward']} "
            f"actor={row['actor_loss']} critic={row['critic_loss']} "
            f"ent={row['entropy']} time={row['time_s']}s"
        )

        self._write(row)

    def _log_eval(self, ep, stats):
        print(
            f"[EVAL {ep}] "
            f"adv={stats['eval_adv_mean']:.3f}±{stats['eval_adv_std']:.3f} "
            f"good={stats['eval_good_mean']:.3f}±{stats['eval_good_std']:.3f}"
        )
        row = {"episode": ep, "type": "eval",
               **{k: f"{v:.4f}" for k, v in stats.items()}}
        self._write(row)

    # ──────────────────────────────────────────────────────────
    # CSV
    # ──────────────────────────────────────────────────────────

    def _write(self, row):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=list(row.keys()),
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()
