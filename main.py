import argparse

from config import get_config
from experiment import train_one_seed



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default=None, choices=["ippo", "mappo"])
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total_episodes", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()



def main():
    cfg = get_config()
    args = parse_args()

    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    train_one_seed(cfg)


if __name__ == "__main__":
    main()
