"""Quick check normal adversary obs layout to confirm ablation dim slicing."""
from mpe2 import simple_world_comm_v3
import numpy as np

env = simple_world_comm_v3.parallel_env(
    num_good=2, num_adversaries=4, num_obstacles=1,
    num_food=2, max_cycles=50, num_forests=2,
    continuous_actions=False,
)
obs, _ = env.reset(seed=42)

normal_adv = "adversary_0"
print(f"Normal adv obs dim: {obs[normal_adv].shape}")
o = obs[normal_adv]
print(f"  [0:2]    self_vel:        {o[0:2]}")
print(f"  [2:4]    self_pos:        {o[2:4]}")
print(f"  [4:6]    landmark_rel:    {o[4:6]}")
print(f"  [6:10]   food_rel:        {o[6:10]}")
print(f"  [10:14]  forest_rel:      {o[10:14]}")
print(f"  [14:24]  other_pos (5):   {o[14:24]}")
print(f"  [24:28]  good_vel (2):    {o[24:28]}")
print(f"  [28:30]  self_in_forest:  {o[28:30]}")
print(f"  [30:34]  leader_comm:     {o[30:34]}")
env.close()
