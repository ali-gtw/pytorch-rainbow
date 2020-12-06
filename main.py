import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import time, os
import wandb

from common.utils import create_log_dir, print_args, set_global_seeds
from common.wrappers import make_atari, wrap_atari_dqn
from arguments import get_args
from train import train
from test import test

def main():
    args = get_args()
    print_args(args)

    log_dir = create_log_dir(args)
    wandb.init(project=args.wandb_project,
     name=args.wandb_name,
     notes=args.wandb_notes,
     config=args)

    env = make_atari(args.env)
    env = wrap_atari_dqn(env, args)

    set_global_seeds(args.seed)
    env.seed(args.seed)

    if args.evaluate:
        test(env, args)
        env.close()
        return

    train(env, args)

    env.close()


if __name__ == "__main__":
    main()
