from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    critic_input: np.ndarray
    action: np.ndarray | int
    log_prob: float
    reward: float
    done: float
    value: float
    next_critic_input: np.ndarray


class PPOBuffer:
    def __init__(self):
        self.storage: list[Transition] = []

    def add(self, transition: Transition) -> None:
        self.storage.append(transition)

    def clear(self) -> None:
        self.storage.clear()

    def __len__(self) -> int:
        return len(self.storage)
