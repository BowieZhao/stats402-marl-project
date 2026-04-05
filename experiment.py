from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from envs import make_env, get_env_dims
from metrics import summarize_episode
from policies import PPOBuffer, Transition, build_policy
from utils import CSVLogger, ensure_dir, save_config, set_seed



def collect_rollout(env, policy, cfg):
    buffer = PPOBuffer()
    rollout_episode_metrics = []

    for _ in range(cfg.rollout_episodes):
        obs_dict, infos = env.reset()
        done_dict = {agent: False for agent in env.possible_agents}

        reward_history = []
        obs_history = []
        info_history = []

        while not all(done_dict.values()):
            action_dict = {}
            log_prob_dict = {}
            value_dict = {}
            critic_input_dict = {}

            for agent in env.possible_agents:
                critic_input = policy.get_critic_input(obs_dict, agent, env.possible_agents)
                action, log_prob, value = policy.select_action(obs_dict[agent], critic_input)
                action_dict[agent] = action
                log_prob_dict[agent] = log_prob
                value_dict[agent] = value
                critic_input_dict[agent] = critic_input

            next_obs_dict, reward_dict, done_dict, infos, terminations, truncations = env.step(action_dict)

            reward_history.append(reward_dict)
            obs_history.append(obs_dict)
            info_history.append(infos)

            for agent in env.possible_agents:
                next_critic_input = policy.get_critic_input(next_obs_dict, agent, env.possible_agents)
                buffer.add(
                    Transition(
                        obs=np.asarray(obs_dict[agent], dtype=np.float32),
                        critic_input=np.asarray(critic_input_dict[agent], dtype=np.float32),
                        action=int(action_dict[agent]),
                        log_prob=float(log_prob_dict[agent]),
                        reward=float(reward_dict[agent]),
                        done=float(done_dict[agent]),
                        value=float(value_dict[agent]),
                        next_critic_input=np.asarray(next_critic_input, dtype=np.float32),
                    )
                )

            obs_dict = next_obs_dict

        rollout_episode_metrics.append(summarize_episode(reward_history, obs_history, info_history))

    mean_metrics = {}
    if rollout_episode_metrics:
        keys = rollout_episode_metrics[0].keys()
        for key in keys:
            mean_metrics[key] = float(np.mean([m[key] for m in rollout_episode_metrics]))
    return buffer, mean_metrics


@np.no_grad if hasattr(np, 'no_grad') else (lambda f: f)
def _noop():
    return None


def evaluate(policy, cfg):
    env = make_env(
        cfg.env_name,
        max_cycles=cfg.max_cycles,
        seed=cfg.seed,
        n_agents=cfg.n_agents,
        local_ratio=cfg.local_ratio,
        continuous_actions=cfg.continuous_actions,
        render_mode=cfg.render_mode,
    )

    episode_metrics = []
    for _ in range(cfg.eval_episodes):
        obs_dict, infos = env.reset()
        done_dict = {agent: False for agent in env.possible_agents}

        reward_history = []
        obs_history = []
        info_history = []

        while not all(done_dict.values()):
            action_dict = {}
            for agent in env.possible_agents:
                critic_input = policy.get_critic_input(obs_dict, agent, env.possible_agents)
                action, _, _ = policy.select_action(obs_dict[agent], critic_input)
                action_dict[agent] = action

            obs_dict, reward_dict, done_dict, infos, terminations, truncations = env.step(action_dict)
            reward_history.append(reward_dict)
            obs_history.append(obs_dict)
            info_history.append(infos)

        episode_metrics.append(summarize_episode(reward_history, obs_history, info_history))

    env.close()
    out = {}
    if episode_metrics:
        keys = episode_metrics[0].keys()
        for key in keys:
            out[f"eval_{key}"] = float(np.mean([m[key] for m in episode_metrics]))
    return out



def train_one_seed(cfg):
    set_seed(cfg.seed)
    ensure_dir(cfg.model_dir)
    ensure_dir(cfg.log_dir)

    env = make_env(
        cfg.env_name,
        max_cycles=cfg.max_cycles,
        seed=cfg.seed,
        n_agents=cfg.n_agents,
        local_ratio=cfg.local_ratio,
        continuous_actions=cfg.continuous_actions,
        render_mode=cfg.render_mode,
    )
    obs_dim, action_dim, global_state_dim, _ = get_env_dims(env)
    policy = build_policy(cfg.algo, obs_dim, action_dim, global_state_dim, cfg)

    logger = CSVLogger(os.path.join(cfg.log_dir, f"{cfg.run_name}_{cfg.algo}_{cfg.seed}.csv"))
    save_config(cfg, os.path.join(cfg.log_dir, f"{cfg.run_name}_{cfg.algo}_{cfg.seed}_config.json"))

    for episode in range(1, cfg.total_episodes + 1):
        buffer, rollout_metrics = collect_rollout(env, policy, cfg)
        train_metrics = policy.update(buffer)

        row = {
            "episode": episode,
            "algo": cfg.algo,
            "seed": cfg.seed,
            **rollout_metrics,
            **train_metrics,
        }

        if episode % cfg.eval_every == 0 or episode == 1:
            eval_metrics = evaluate(policy, cfg)
            row.update(eval_metrics)
            print(
                f"[Eval] ep={episode} algo={cfg.algo} "
                f"train_reward={rollout_metrics.get('episode_reward', 0.0):.3f} "
                f"eval_reward={eval_metrics.get('eval_episode_reward', 0.0):.3f}"
            )
        else:
            print(
                f"[Train] ep={episode} algo={cfg.algo} "
                f"reward={rollout_metrics.get('episode_reward', 0.0):.3f}"
            )

        logger.log(row)

        if episode % cfg.save_every == 0 or episode == cfg.total_episodes:
            model_path = os.path.join(cfg.model_dir, f"{cfg.run_name}_{cfg.algo}_{cfg.seed}_ep{episode}.pt")
            policy.save(model_path)

    env.close()
    return policy
