from __future__ import annotations

import numpy as np



def compute_episode_reward(reward_history: list[dict]) -> float:
    total = 0.0
    for step_rewards in reward_history:
        total += float(sum(step_rewards.values()))
    return total



def compute_agent_reward_means(reward_history: list[dict]) -> dict:
    if not reward_history:
        return {}
    agents = reward_history[0].keys()
    out = {}
    for agent in agents:
        out[agent] = float(np.mean([step[agent] for step in reward_history]))
    return out



def compute_collision_rate(info_history: list[dict]) -> float:
    values = []
    for step_info in info_history:
        for _, info in step_info.items():
            if isinstance(info, dict) and "collisions" in info:
                values.append(float(info["collisions"]))
    return float(np.mean(values)) if values else 0.0



def compute_coverage_efficiency(obs_history: list[dict]) -> float:
    # Placeholder metric for Stage 1.
    # If you later expose landmark occupancy from env internals, replace this.
    # For now, return a proxy based on whether all agents remain active.
    if not obs_history:
        return 0.0
    per_step = []
    for step_obs in obs_history:
        active = sum(1 for _, obs in step_obs.items() if obs is not None)
        total = max(len(step_obs), 1)
        per_step.append(active / total)
    return float(np.mean(per_step))



def summarize_episode(reward_history: list[dict], obs_history: list[dict], info_history: list[dict]) -> dict:
    return {
        "episode_reward": compute_episode_reward(reward_history),
        "coverage_efficiency": compute_coverage_efficiency(obs_history),
        "collision_rate": compute_collision_rate(info_history),
    }
