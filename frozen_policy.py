"""
Frozen good policy - loads the good agent actor from a trained MAPPO checkpoint
and provides action selection for inference only. Used to hold the good-agent
side fixed during adversary-side experiments, ensuring fair comparison across
different adversary algorithms.

If the env is in continuous-action mode (e.g. for DDPG), the policy outputs
the discrete action sample one-hot encoded as a Box(0,1,(action_dim,)) vector,
which the underlying mpe2 env interprets correctly.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import Dict


def _mlp(in_dim: int, out_dim: int, hidden: int = 128) -> nn.Sequential:
    """Must match the structure in mappo.py _mlp."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class FrozenActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = _mlp(obs_dim, action_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrozenGoodPolicy:
    """
    Loads a frozen good-agent actor from a MAPPO checkpoint directory.
    Expected layout: <path>/good.pt containing {"actor": state_dict, ...}.

    If continuous=True, .act() returns a one-hot float vector
    (since mpe2 with continuous_actions=True accepts Box(0, 1, (action_dim,))).
    Otherwise returns an int.
    """

    def __init__(self, checkpoint_path: str, obs_dim: int, action_dim: int,
                 hidden: int = 128, device: str = "cpu",
                 deterministic: bool = False, continuous: bool = False):
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.continuous = continuous
        self.action_dim = action_dim

        self.actor = FrozenActor(obs_dim, action_dim, hidden).to(self.device)

        good_ckpt_path = os.path.join(checkpoint_path, "good.pt")
        if not os.path.exists(good_ckpt_path):
            raise FileNotFoundError(
                f"Could not find good.pt in checkpoint directory: {checkpoint_path}"
            )

        ckpt = torch.load(good_ckpt_path, map_location=self.device,
                          weights_only=False)
        actor_state = ckpt["actor"]
        self.actor.load_state_dict(actor_state)
        self.actor.eval()

        print(f"[FrozenGoodPolicy] loaded from {good_ckpt_path}")
        print(f"  obs_dim={obs_dim}, action_dim={action_dim}, "
              f"deterministic={deterministic}, continuous={continuous}")

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        """Sample (or argmax) an action for a single good agent.
        Returns int if discrete, np.float32 array if continuous."""
        if getattr(self, "_force_random", False):
            idx = int(np.random.randint(self.action_dim))
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            logits = self.actor(obs_t)
            if self.deterministic:
                idx = int(logits.argmax(dim=-1).item())
            else:
                dist = Categorical(logits=logits)
                idx = int(dist.sample().item())

        if self.continuous:
            # one-hot vector — mpe2 continuous mode accepts Box(0,1,(action_dim,))
            v = np.zeros(self.action_dim, dtype=np.float32)
            v[idx] = 1.0
            return v
        return idx

    def act_batch(self, obs_dict: Dict[str, np.ndarray], good_agent_names):
        out = {}
        for name in good_agent_names:
            if name in obs_dict:
                out[name] = self.act(obs_dict[name])
        return out
