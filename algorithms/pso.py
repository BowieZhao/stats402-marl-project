from __future__ import annotations

import os
import numpy as np
import random


# =========================
# Simple PSO Particle
# =========================
class Particle:
    def __init__(self, dim):
        self.position = np.random.uniform(-1, 1, size=dim)
        self.velocity = np.zeros(dim)

        self.best_position = self.position.copy()
        self.best_score = -float("inf")


# =========================
# PSO Agent (single)
# =========================
class SinglePSO:
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.num_particles = 10
        self.particles = [Particle(action_dim) for _ in range(self.num_particles)]

        self.global_best = np.zeros(action_dim)
        self.global_best_score = -float("inf")

        # PSO params
        self.w = 0.5
        self.c1 = 1.0
        self.c2 = 1.0

    def evaluate(self, obs, action_vec):
        """
        简单 heuristic:
        用 action 和 observation 的“对齐程度”作为 pseudo reward
        （只是为了让PSO有个优化目标）
        """
        obs = np.asarray(obs)
        action_vec = np.asarray(action_vec)

        # 截断到同维度
        dim = min(len(obs), len(action_vec))
        return float(np.dot(obs[:dim], action_vec[:dim]))

    def optimize(self, obs):
        for p in self.particles:

            score = self.evaluate(obs, p.position)

            # update personal best
            if score > p.best_score:
                p.best_score = score
                p.best_position = p.position.copy()

            # update global best
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best = p.position.copy()

        # update velocity + position
        for p in self.particles:
            r1, r2 = random.random(), random.random()

            p.velocity = (
                self.w * p.velocity
                + self.c1 * r1 * (p.best_position - p.position)
                + self.c2 * r2 * (self.global_best - p.position)
            )

            p.position += p.velocity

    def act(self, obs):
        self.optimize(obs)

        # 把连续向量 → 离散动作
        return int(np.argmax(self.global_best))


# =========================
# Multi-Agent Wrapper
# =========================
class PSOMultiAgent:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

        self.agents = {}

        # 👉 VERY IMPORTANT（给 Experiment 用）
        self.adv_agents = []
        self.good_agents = []

        for name in env.possible_agents:
            spec = env.get_agent_spec(name)

            self.agents[name] = SinglePSO(
                obs_dim=spec.obs_dim,
                action_dim=spec.action_dim
            )

            # 分组（避免你之前报错）
            if "adversary" in name:
                self.adv_agents.append(name)
            else:
                self.good_agents.append(name)

    # =========================
    def select_actions(self, obs, env=None, explore=True):
        actions = {}

        for name, ob in obs.items():
            actions[name] = self.agents[name].act(ob)

        return actions

    # =========================
    # 下面这些是接口兼容（不做任何事）
    # =========================
    def observe(self, transition):
        pass

    def update(self, global_step=None):
        return {}

    def end_episode(self):
        pass

    def save(self, path):
        os.makedirs(path, exist_ok=True)

    def load(self, path):
        pass