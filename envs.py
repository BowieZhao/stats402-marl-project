from __future__ import annotations

from typing import Any
import numpy as np


def _import_simple_spread():
    try:
        from pettingzoo.mpe import simple_spread_v3
        return simple_spread_v3
    except Exception:
        from mpe2 import simple_spread_v3  # type: ignore
        return simple_spread_v3


class ParallelEnvWrapper:
    def __init__(self, env: Any):
        self.env = env
        self.possible_agents = list(env.possible_agents)
        self.n_agents = len(self.possible_agents)

    def reset(self, seed: int | None = None):
        out = self.env.reset(seed=seed)
        if isinstance(out, tuple):
            obs, infos = out
        else:
            obs, infos = out, {agent: {} for agent in self.possible_agents}
        return obs, infos

    def step(self, actions: dict):
        out = self.env.step(actions)
        if len(out) == 5:
            obs, rewards, terminations, truncations, infos = out
        else:
            raise RuntimeError("Unexpected env.step output format.")

        dones = {
            agent: bool(terminations.get(agent, False) or truncations.get(agent, False))
            for agent in self.possible_agents
        }
        return obs, rewards, dones, infos, terminations, truncations

    def close(self):
        self.env.close()



def make_env(env_name: str, max_cycles: int = 40, seed: int = 0, **kwargs):
    env_name = env_name.lower()
    if env_name == "simple_spread":
        module = _import_simple_spread()
        env = module.parallel_env(
            N=kwargs.get("n_agents", 3),
            local_ratio=kwargs.get("local_ratio", 0.5),
            max_cycles=max_cycles,
            continuous_actions=kwargs.get("continuous_actions", False),
            render_mode=kwargs.get("render_mode", None),
        )
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    wrapped = ParallelEnvWrapper(env)
    wrapped.reset(seed=seed)
    return wrapped



def get_env_dims(env: ParallelEnvWrapper):
    first_agent = env.possible_agents[0]
    obs_space = env.env.observation_space(first_agent)
    act_space = env.env.action_space(first_agent)

    if not hasattr(obs_space, "shape") or obs_space.shape is None:
        raise ValueError("Observation space must have a shape.")
    obs_dim = int(np.prod(obs_space.shape))

    if hasattr(act_space, "n"):
        action_dim = int(act_space.n)
        is_discrete = True
    else:
        raise ValueError("This starter code currently supports discrete action spaces only.")

    global_state_dim = obs_dim * len(env.possible_agents)
    return obs_dim, action_dim, global_state_dim, is_discrete
