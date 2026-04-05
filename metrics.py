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


def compute_collision_rate_from_state(
    state_history: list[dict],
) -> float:
    """
    Average number of pairwise agent-agent collisions per timestep.
    Two agents collide if distance < size_i + size_j.
    """
    if not state_history:
        return 0.0

    collisions_per_step = []

    for state in state_history:
        agent_positions = state["agent_positions"]
        agent_sizes = state["agent_sizes"]
        agents = list(agent_positions.keys())

        num_collisions = 0
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                ai, aj = agents[i], agents[j]
                pi = agent_positions[ai]
                pj = agent_positions[aj]
                dist = np.linalg.norm(pi - pj)
                threshold = agent_sizes[ai] + agent_sizes[aj]
                if dist < threshold:
                    num_collisions += 1

        collisions_per_step.append(num_collisions)

    return float(np.mean(collisions_per_step))


def compute_coverage_efficiency_from_state(
    state_history: list[dict],
) -> float:
    """
    Coverage Efficiency =
    (# covered landmarks / total landmarks), averaged over timesteps.

    A landmark is counted as covered if at least one agent is within
    landmark.size + agent.size of that landmark.
    """
    if not state_history:
        return 0.0

    coverage_values = []

    for state in state_history:
        agent_positions = state["agent_positions"]
        agent_sizes = state["agent_sizes"]
        landmark_positions = state["landmark_positions"]
        landmark_sizes = state["landmark_sizes"]

        if len(landmark_positions) == 0:
            coverage_values.append(0.0)
            continue

        covered = 0
        for l_idx, l_pos in enumerate(landmark_positions):
            landmark_covered = False
            for agent_name, a_pos in agent_positions.items():
                dist = np.linalg.norm(a_pos - l_pos)
                threshold = landmark_sizes[l_idx] + agent_sizes[agent_name]
                if dist < threshold:
                    landmark_covered = True
                    break
            if landmark_covered:
                covered += 1

        coverage_values.append(covered / len(landmark_positions))

    return float(np.mean(coverage_values))


def summarize_episode(
    reward_history: list[dict],
    state_history: list[dict],
) -> dict:
    return {
        "episode_reward": compute_episode_reward(reward_history),
        "coverage_efficiency": compute_coverage_efficiency_from_state(state_history),
        "collision_rate": compute_collision_rate_from_state(state_history),
    }
