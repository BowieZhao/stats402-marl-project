"""
PSO baseline for simple_world_comm.

Design: each particle encodes the flattened weights of a small policy network.
Fitness = average episode reward of that particle's policy.
We update particle velocities toward personal and global best policies.

This is the "neuroevolution via PSO" formulation. It is intentionally kept simple
with a small network and few particles so it runs in reasonable time.
It is expected to be substantially weaker than MAPPO/DQN — that is the point of
a baseline.

Interface note: PSO does not use step-level observation/transition learning the
way PPO/DQN do. observe() and update() are no-ops; instead, the main loop runs
a fitness evaluation episode per particle via select_actions() and we update
the swarm at end_episode() boundaries.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════
# Small policy network (same arch shared across particles/agents)
# ═══════════════════════════════════════════════════════════
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([p.detach().cpu().numpy().ravel()
                               for p in self.parameters()])

    def set_flat_params(self, flat: np.ndarray):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            chunk = flat[idx:idx + n]
            p.data.copy_(torch.from_numpy(chunk.reshape(p.shape)).float())
            idx += n


# ═══════════════════════════════════════════════════════════
# PSO particle
# ═══════════════════════════════════════════════════════════
class Particle:
    def __init__(self, dim, scale=0.5):
        self.position = np.random.randn(dim).astype(np.float32) * scale
        self.velocity = np.zeros(dim, dtype=np.float32)
        self.best_position = self.position.copy()
        self.best_fitness = -float("inf")


# ═══════════════════════════════════════════════════════════
# PSO agent wrapper
# ═══════════════════════════════════════════════════════════
class PSOAgent:
    """
    One swarm per role (leader / adversary / good). Within a role, all agents
    share the same policy (same flat weight vector) per particle.
    This matches the parameter-sharing convention used by MAPPO.
    """

    def __init__(self, cfg, env):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Identify roles
        self.adv_agents = [a for a in env.possible_agents if "adversary" in a]
        self.good_agents = [a for a in env.possible_agents if a.startswith("agent_")]
        self.leader_agents = [a for a in env.possible_agents if "leadadversary" in a]
        self.normal_adv_agents = [a for a in self.adv_agents if a not in self.leader_agents]

        # Build one reference policy per role (for shape info)
        self.role_policies: Dict[str, PolicyNet] = {}
        if self.leader_agents:
            spec = env.get_agent_spec(self.leader_agents[0])
            self.role_policies["leader"] = PolicyNet(spec.obs_dim, spec.action_dim).to(self.device)
        if self.normal_adv_agents:
            spec = env.get_agent_spec(self.normal_adv_agents[0])
            self.role_policies["adversary"] = PolicyNet(spec.obs_dim, spec.action_dim).to(self.device)
        if self.good_agents:
            spec = env.get_agent_spec(self.good_agents[0])
            self.role_policies["good"] = PolicyNet(spec.obs_dim, spec.action_dim).to(self.device)

        # PSO swarms per role
        self.num_particles = 8
        self.swarms: Dict[str, List[Particle]] = {}
        self.global_best: Dict[str, np.ndarray] = {}
        self.global_best_fitness: Dict[str, float] = {}
        for role, net in self.role_policies.items():
            dim = net.num_params()
            self.swarms[role] = [Particle(dim) for _ in range(self.num_particles)]
            self.global_best[role] = self.swarms[role][0].position.copy()
            self.global_best_fitness[role] = -float("inf")

        # PSO hyperparams
        self.w_inertia = 0.5
        self.c1 = 1.0
        self.c2 = 1.0
        self.vel_clip = 2.0

        # Current episode tracking
        self.current_particle_idx = 0
        self.current_episode_rewards: Dict[str, float] = {}

        # Load current particle weights into the networks
        self._load_particle(self.current_particle_idx)

    # ── Weight loading ──────────────────────────────────
    def _load_particle(self, idx: int):
        """Load the idx-th particle's weights into each role's policy network."""
        for role, swarm in self.swarms.items():
            self.role_policies[role].set_flat_params(swarm[idx].position)

    def _get_role(self, agent_name: str) -> str:
        if "leadadversary" in agent_name:
            return "leader"
        if "adversary" in agent_name:
            return "adversary"
        return "good"

    # ── Required interface ──────────────────────────────
    def select_actions(self, obs, env=None, explore=True):
        """Argmax over the current particle's policy logits."""
        actions = {}
        for name, ob in obs.items():
            role = self._get_role(name)
            if role not in self.role_policies:
                continue
            obs_t = torch.tensor(ob, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.role_policies[role](obs_t)
            actions[name] = int(logits.argmax(dim=-1).item())
        return actions

    def observe(self, transition):
        """Track per-agent rewards for fitness computation."""
        for a, r in transition["rewards"].items():
            self.current_episode_rewards[a] = \
                self.current_episode_rewards.get(a, 0.0) + float(r)

    def end_episode(self):
        """
        Evaluate current particle, update its personal/global best,
        then move to the next particle. When all particles have been
        evaluated, run the PSO velocity/position update.
        """
        idx = self.current_particle_idx

        # Compute per-role fitness (team mean reward)
        fitness_per_role = {}
        if self.adv_agents:
            fitness_per_role["leader"] = fitness_per_role["adversary"] = \
                float(np.mean([self.current_episode_rewards.get(a, 0.0)
                               for a in self.adv_agents]))
        if self.good_agents:
            fitness_per_role["good"] = \
                float(np.mean([self.current_episode_rewards.get(a, 0.0)
                               for a in self.good_agents]))

        # Update personal and global best for each role
        for role, swarm in self.swarms.items():
            fit = fitness_per_role.get(role, 0.0)
            particle = swarm[idx]
            if fit > particle.best_fitness:
                particle.best_fitness = fit
                particle.best_position = particle.position.copy()
            if fit > self.global_best_fitness[role]:
                self.global_best_fitness[role] = fit
                self.global_best[role] = particle.position.copy()

        # Reset episode reward tracking
        self.current_episode_rewards = {}

        # Move to next particle
        self.current_particle_idx = (self.current_particle_idx + 1) % self.num_particles

        # If we've evaluated all particles, update the swarm
        if self.current_particle_idx == 0:
            self._update_swarm()

        # Load the next particle for the upcoming episode
        self._load_particle(self.current_particle_idx)

    def _update_swarm(self):
        """PSO velocity and position update across all particles."""
        for role, swarm in self.swarms.items():
            g_best = self.global_best[role]
            for p in swarm:
                r1 = np.random.rand(*p.velocity.shape).astype(np.float32)
                r2 = np.random.rand(*p.velocity.shape).astype(np.float32)
                p.velocity = (
                    self.w_inertia * p.velocity
                    + self.c1 * r1 * (p.best_position - p.position)
                    + self.c2 * r2 * (g_best - p.position)
                )
                np.clip(p.velocity, -self.vel_clip, self.vel_clip, out=p.velocity)
                p.position = p.position + p.velocity

    def update(self, global_step=None):
        """PSO has no gradient update; return fitness metrics for logging."""
        logs = {}
        for role, fit in self.global_best_fitness.items():
            logs[f"{role}/global_best_fitness"] = fit
        logs["current_particle"] = self.current_particle_idx
        return logs

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        # Save the global-best policy weights for each role
        torch.save(
            {role: {"global_best": self.global_best[role],
                    "fitness": self.global_best_fitness[role]}
             for role in self.role_policies.keys()},
            os.path.join(path, "pso.pt")
        )

    def load(self, path):
        fp = path if path.endswith(".pt") else os.path.join(path, "pso.pt")
        state = torch.load(fp, map_location="cpu", weights_only=False)
        for role, data in state.items():
            if role in self.role_policies:
                self.global_best[role] = data["global_best"]
                self.global_best_fitness[role] = data["fitness"]
                self.role_policies[role].set_flat_params(data["global_best"])
