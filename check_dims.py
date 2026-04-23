"""
Check observation dimensions when num_good changes from 2 to 3.
This tells us if we need an obs adapter for zero-shot generalization testing.
"""

from mpe2 import simple_world_comm_v3

print("=" * 50)
print("Comparing obs dims: num_good=2 vs num_good=3")
print("=" * 50)

for num_good in [2, 3]:
    env = simple_world_comm_v3.parallel_env(
        num_good=num_good,
        num_adversaries=4,
        num_obstacles=1,
        num_food=2,
        max_cycles=50,
        num_forests=2,
        continuous_actions=False,
    )
    obs, _ = env.reset(seed=42)

    print(f"\nnum_good = {num_good}")
    print(f"  agents: {list(env.possible_agents)}")
    for agent in env.possible_agents:
        ob = obs[agent]
        act_space = env.action_space(agent)
        act_n = act_space.n if hasattr(act_space, "n") else act_space.shape
        print(f"    {agent:<20s}  obs_dim={len(ob):<4d}  act_dim={act_n}")

    env.close()

print()
print("=" * 50)
print("INTERPRETATION")
print("=" * 50)
print("""
If adversary obs_dim differs between num_good=2 and num_good=3:
  → checkpoints trained on 2 good CANNOT be loaded directly into 3 good env
  → we need an obs adapter for zero-shot eval

If obs_dim is the same:
  → generalization test is straightforward, just load and evaluate
""")
