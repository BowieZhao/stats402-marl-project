from __future__ import annotations

import argparse
import pandas as pd
import matplotlib.pyplot as plt



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--x", type=str, default="episode")
    parser.add_argument("--y", type=str, default="eval_adversary_episode_reward")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df[args.x], df[args.y])
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title(f"{args.y} vs {args.x}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
