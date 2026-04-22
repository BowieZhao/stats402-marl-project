from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import torch


class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        self._fieldnames: list[str] | None = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        write_header = not os.path.exists(self.path)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def to_tensor(x: Any, device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)



def linear_schedule(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    frac = min(step / decay_steps, 1.0)
    return start + frac * (end - start)
