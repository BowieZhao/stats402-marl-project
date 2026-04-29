"""
Verify the decoding of leader's joint action a in [0, 19].
We send each action and inspect the resulting leader_comm in normal adv obs.
The leader_comm in obs is a one-hot of the say index, so we can read the
true say_idx for each action a.
"""

from mpe2 import simple_world_comm_v3
import numpy as np

env = simple_world_comm_v3.parallel_env(
    num_good=2, num_adversaries=4, num_obstacles=1,
    num_food=2, max_cycles=50, num_forests=2,
    continuous_actions=False,
)
obs, _ = env.reset(seed=42)

leader = "leadadversary_0"
adv = "adversary_0"

print(f"Leader action_space: {env.action_space(leader)}")
print(f"Normal adv obs dim: {obs[adv].shape}")
print()
print(f"  a  | leader_comm[30:34] from adv obs | inferred say_idx")
print(f"-----+----------------------------------+------------------")

for a in range(20):
    obs, _ = env.reset(seed=42)
    actions = {agent: 0 for agent in env.agents}  # all noop/say_0
    actions[leader] = a
    obs_next, _, _, _, _ = env.step(actions)

    leader_comm = obs_next[adv][30:34]
    say_idx = int(np.argmax(leader_comm))
    move_idx = int(actions[leader] % 5)  # not what we're testing
    print(f"  {a:2d} | {leader_comm}  | say_idx = {say_idx}")

env.close()
print()
print("If say_idx = a // 5: encoding is [say, move] (say is high)")
print("If say_idx = a % 4:  encoding is [move, say] (say is low)")
