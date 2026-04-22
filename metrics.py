import numpy as np


def summarize_episode(reward_history, state_history=None):

    if not reward_history:
        return {
            "episode_reward": 0.0,
            "adversary_episode_reward": 0.0,
            "good_episode_reward": 0.0,
            "capture_rate": 0.0,
            "capture_time": 0.0,
            "coordination_delay": 0.0,
            "reward_std": 0.0,
        }

    total = 0.0
    adv = 0.0
    good = 0.0

    per_step_totals = []

    for step in reward_history:

        step_total = 0.0

        for k, v in step.items():

            v = float(v)
            total += v
            step_total += v

            if "adversary" in k or "lead" in k:
                adv += v
            elif "agent" in k:
                good += v

        per_step_totals.append(step_total)

    episode_reward = total
    adv_reward = adv
    good_reward = good

    # ===== proper metrics =====

    capture_rate = 1.0 if good_reward > adv_reward else 0.0
    capture_time = len(reward_history)

    # real proxy of instability / coordination
    coordination_delay = float(np.std(per_step_totals)) if len(per_step_totals) > 1 else 0.0

    reward_std = float(np.std(per_step_totals)) if len(per_step_totals) > 1 else 0.0

    return {
        "episode_reward": episode_reward,
        "adversary_episode_reward": adv_reward,
        "good_episode_reward": good_reward,

        "capture_rate": capture_rate,
        "capture_time": capture_time,
        "coordination_delay": coordination_delay,
        "reward_std": reward_std,
    }