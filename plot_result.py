import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/logs/tag_debug_mappo_42.csv")
df = pd.read_csv("outputs/logs/tag_debug_mappo_42.csv")

eval_df = df.dropna(subset=["eval_episode_reward"])

eval_df = eval_df.sort_values("episode")

episodes = eval_df["episode"]

reward = eval_df["eval_episode_reward"]
capture_time = eval_df["eval_capture_time"]


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(episodes, reward, marker='o')
plt.title("Eval Reward")
plt.xlabel("Episode")

plt.subplot(1,2,2)
plt.plot(episodes, capture_time, marker='o')
plt.title("Eval Capture Time")
plt.xlabel("Episode")

plt.tight_layout()
plt.show()