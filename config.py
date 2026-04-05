from dataclasses import dataclass, asdict


@dataclass
class Config:
    # experiment
    env_name: str = "simple_spread"
    algo: str = "mappo"  # choices: ippo, mappo
    seed: int = 42
    total_episodes: int = 800
    max_cycles: int = 40
    eval_every: int = 50
    eval_episodes: int = 10
    save_every: int = 100
    device: str = "cpu"
    run_name: str = "debug_run"
    model_dir: str = "./outputs/models"
    log_dir: str = "./outputs/logs"

    # environment
    n_agents: int = 3
    n_landmarks: int = 3
    local_ratio: float = 0.5
    continuous_actions: bool = False
    render_mode: str | None = None

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    rollout_episodes: int = 8

    # optimization
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # network
    hidden_dim: int = 128
    shared_actor: bool = True
    shared_critic: bool = True



def get_config() -> Config:
    return Config()



def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)
