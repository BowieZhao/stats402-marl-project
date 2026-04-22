from __future__ import annotations

import argparse
import pandas as pd



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", nargs="+")
    parser.add_argument("--metric", type=str, default="eval_adversary_episode_reward")
    args = parser.parse_args()

    rows = []
    for path in args.csv_paths:
        df = pd.read_csv(path)
        algo = str(df["algo"].iloc[0]) if "algo" in df.columns else path
        value = df[args.metric].dropna().iloc[-1] if args.metric in df.columns else float("nan")
        rows.append({"algo": algo, "metric": args.metric, "value": value, "path": path})

    out = pd.DataFrame(rows)
    print(out)


if __name__ == "__main__":
    main()
