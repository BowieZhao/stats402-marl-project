import argparse

from config import get_config
from envs import make_env, get_env_dims
from policies import build_policy
from utils import set_seed


def visualize(model_path, algo, episodes=2):
    cfg = get_config()
    cfg.algo = algo

    set_seed(cfg.seed)

    env = make_env(
        cfg.env_name,
        max_cycles=cfg.max_cycles,
        seed=cfg.seed,
        n_agents=cfg.n_agents,
        local_ratio=cfg.local_ratio,
        continuous_actions=cfg.continuous_actions,
        render_mode="human",
    )

    obs_dim, action_dim, global_state_dim, _ = get_env_dims(env)
    policy = build_policy(cfg.algo, obs_dim, action_dim, global_state_dim, cfg)

    policy.load(model_path)

    for ep in range(episodes):
        obs_dict, infos = env.reset()
        done_dict = {agent: False for agent in env.possible_agents}
        total_reward = 0.0

        while not all(done_dict.values()):
            action_dict = {}
            for agent in env.possible_agents:
                critic_input = policy.get_critic_input(obs_dict, agent, env.possible_agents)
                action, _, _ = policy.select_action(obs_dict[agent], critic_input)
                action_dict[agent] = action

            obs_dict, reward_dict, done_dict, infos, terminations, truncations = env.step(action_dict)
            total_reward += sum(reward_dict.values())

        print(f"[Visualize] Episode {ep+1}, reward = {total_reward:.3f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True, choices=["ippo", "mappo"])
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()

    visualize(
        model_path=args.model_path,
        algo=args.algo,
        episodes=args.episodes,
    )