import os
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
        state_history = []

        while not all(done_dict.values()):
            action_dict = {}
            log_prob_dict = {}
            value_dict = {}
            critic_input_dict = {}

            state_history.append(env.get_world_state())

            for agent in env.possible_agents:
                critic_input = policy.get_critic_input(obs_dict, agent, env.possible_agents)
                action, log_prob, value = policy.select_action(obs_dict[agent], critic_input)

                action_dict[agent] = action
                log_prob_dict[agent] = log_prob
                value_dict[agent] = value
                critic_input_dict[agent] = critic_input

            next_obs_dict, reward_dict, done_dict, infos, terminations, truncations = env.step(action_dict)

            # ===== 🔥 使用 config 控制 collision penalty =====
            adjusted_reward_dict = {}
            for agent in env.possible_agents:
                collision = infos.get(agent, {}).get("collisions", 0.0)
                adjusted_reward_dict[agent] = float(reward_dict[agent]) - cfg.collision_penalty * float(collision)
            # ==================================================

            reward_history.append(adjusted_reward_dict)

            for agent in env.possible_agents:
                next_critic_input = policy.get_critic_input(next_obs_dict, agent, env.possible_agents)

                buffer.add(
                    Transition(
                        obs=np.asarray(obs_dict[agent], dtype=np.float32),
                        critic_input=np.asarray(critic_input_dict[agent], dtype=np.float32),
                        action=int(action_dict[agent]),
                        log_prob=float(log_prob_dict[agent]),
                        reward=float(adjusted_reward_dict[agent]),
                        done=float(done_dict[agent]),
                        value=float(value_dict[agent]),
                        next_critic_input=np.asarray(next_critic_input, dtype=np.float32),
                    )
                )

            obs_dict = next_obs_dict

        rollout_episode_metrics.append(summarize_episode(reward_history, state_history))

    mean_metrics = {}
    if rollout_episode_metrics:
        keys = rollout_episode_metrics[0].keys()
        for key in keys:
            mean_metrics[key] = float(np.mean([m[key] for m in rollout_episode_metrics]))

    return buffer, mean_metrics


def evaluate(policy, cfg):
    env = make_env(
        cfg.env_name,
        max_cycles=cfg.max_cycles,
        seed=cfg.seed,
        n_agents=cfg.n_agents,
        local_ratio=cfg.local_ratio,
        continuous_actions=cfg.continuous_actions,
        render_mode=None,
    )

    episode_metrics = []

    for _ in range(cfg.eval_episodes):
        obs_dict, infos = env.reset()
        done_dict = {agent: False for agent in env.possible_agents}

        reward_history = []
        state_history = []

        while not all(done_dict.values()):
            state_history.append(env.get_world_state())

            action_dict = {}
            for agent in env.possible_agents:
                critic_input = policy.get_critic_input(obs_dict, agent, env.possible_agents)
                action, _ = policy.act_deterministic(obs_dict[agent], critic_input)
                action_dict[agent] = action

            obs_dict, reward_dict, done_dict, infos, terminations, truncations = env.step(action_dict)
            reward_history.append(reward_dict)

        episode_metrics.append(summarize_episode(reward_history, state_history))

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

    best_eval_reward = -float("inf")

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

            current_eval = eval_metrics.get("eval_episode_reward", -float("inf"))
            if current_eval > best_eval_reward:
                best_eval_reward = current_eval
                best_model_path = os.path.join(
                    cfg.model_dir,
                    f"{cfg.run_name}_{cfg.algo}_{cfg.seed}_best.pt",
                )
                policy.save(best_model_path)

            print(
                f"[Eval] ep={episode} algo={cfg.algo} "
                f"train_reward={rollout_metrics.get('episode_reward', 0.0):.3f} "
                f"eval_reward={eval_metrics.get('eval_episode_reward', 0.0):.3f} "
                f"coverage={rollout_metrics.get('coverage_efficiency', 0.0):.3f} "
                f"collision={rollout_metrics.get('collision_rate', 0.0):.3f}"
            )
        else:
            print(
                f"[Train] ep={episode} algo={cfg.algo} "
                f"reward={rollout_metrics.get('episode_reward', 0.0):.3f} "
                f"coverage={rollout_metrics.get('coverage_efficiency', 0.0):.3f} "
                f"collision={rollout_metrics.get('collision_rate', 0.0):.3f}"
            )

        logger.log(row)

        if episode % cfg.save_every == 0 or episode == cfg.total_episodes:
            model_path = os.path.join(
                cfg.model_dir,
                f"{cfg.run_name}_{cfg.algo}_{cfg.seed}_ep{episode}.pt",
            )
            policy.save(model_path)

    env.close()
    return policy