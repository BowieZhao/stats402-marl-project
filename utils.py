from __future__ import annotations

import csv
import os
import random
from dataclasses import asdict

import numpy as np
import torch



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)



def obs_dict_to_global_state(obs_dict: dict, agent_order: list[str]) -> np.ndarray:
    parts = []
    for agent in agent_order:
        obs = obs_dict[agent]
        parts.append(np.asarray(obs, dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0)



def to_tensor(x, device: str):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


class CSVLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        ensure_dir(os.path.dirname(csv_path))
        self.header_written = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    def log(self, row: dict):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)



def save_config(cfg, path: str):
    import json
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
