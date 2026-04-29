"""
Frozen good policy - loads the good agent actor from a trained MAPPO checkpoint
and provides action selection for inference only.

Supports two checkpoint formats:
  1. Old format: ckpt is a dict with keys like {"actor": state_dict, ...}
  2. New format: ckpt is the state_dict directly (saved as torch.save(state_dict, ...))
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import Dict


def _mlp(in_dim: int, out_dim: int, hidden: int = 128) -> nn.Sequential:
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

        # Handle both formats
        if isinstance(ckpt, dict) and "actor" in ckpt:
            actor_state = ckpt["actor"]  # old format
        elif isinstance(ckpt, dict) and any(k.startswith("net.") for k in ckpt.keys()):
            actor_state = ckpt  # new format - state_dict directly
        else:
            actor_state = ckpt  # fall through

        try:
            self.actor.load_state_dict(actor_state)
        except Exception as e:
            print(f"[FrozenGoodPolicy] WARNING loading state_dict: {e}")
            print(f"[FrozenGoodPolicy] Trying with strict=False")
            self.actor.load_state_dict(actor_state, strict=False)

        self.actor.eval()

        print(f"[FrozenGoodPolicy] loaded from {good_ckpt_path}")
        print(f"  obs_dim={obs_dim}, action_dim={action_dim}, "
              f"deterministic={deterministic}, continuous={continuous}")

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        logits = self.actor(obs_t)
        if self.deterministic:
            idx = int(logits.argmax(dim=-1).item())
        else:
            dist = Categorical(logits=logits)
            idx = int(dist.sample().item())

        if self.continuous:
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
