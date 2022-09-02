import sys
import argparse
import gym
import torch
import numpy as np
import random
from multiprocessing import Process, Manager

from meta_ppo import Meta_PPO, FeedForwardNN, LossNN


def get_args():
    """
    Description:
        Parses arguments at command line.

    Parameters:
        None

    Return:
        args - the arguments parsed
    """
    parser = argparse.ArgumentParser()

    # can be 'meta_train' or 'meta_test'
    parser.add_argument('--mode', dest='mode', type=str, default='meta_train')
    parser.add_argument('--loss_fn', dest='loss_fn',
                        type=str, default='')   # your loss model filename

    args = parser.parse_args()

    return args


def meta_train(env, agent_num, loss_fn, hyperparameters, mean_returns):
    """
    Meta trains a single agent only, returning its mean return of the last 100 timesteps

    Parameters:
        env - the environment to meta_train on
        hyperparameters - a dict of hyperparameters to use, defined in main

    Return:
        None
    """
    print(f"meta training agent {agent_num}")
    model = Meta_PPO(policy_class=FeedForwardNN,
                     loss_fn=loss_fn, env=env, **hyperparameters)
    mean_returns[agent_num] = model.train(total_timesteps=10_000)


def meta_test(env, hyperparameters, loss_fn):
    """
    Parameters:
        env - the environment to meta_test the policy on
        loss_fn - the shared loss function retrieved from meta-training

    Return:
        None
    """
    print(f"Meta-Testing {loss_fn}")

    # If the loss function is not specified, then exit
    if loss_fn == '':
        print(f"Didn't specify loss function file. Exiting.")
        sys.exit(0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    loss_fn = LossNN(obs_dim, act_dim, mode="testing")
    # strict is set to False here to ignore non-matching keys
    loss_fn.load_state_dict(torch.load(loss_fn), strict=False)

    model = Meta_PPO(policy_class=FeedForwardNN,
                     loss_fn=loss_fn, env=env, **hyperparameters)
    model.train(total_timesteps=1_000_000)  # equivalent to meta-testing


def main(args):
    """
    Parameters:
        args - the arguments parsed from command line

    Return:
        None
    """
    hyperparameters = {
        'timesteps_per_batch': 4096,  # Number of timesteps to run per batch
        'timesteps_per_episode': 256,  # Number of timesteps per episode
        'gamma': 0.95,  # Discount factor to be applied when calculating Rewards-To-Go
        'num_epochs': 15,  # Number of epochs to update actor/critic per iteration
        'alph': 3e-3,  # alpha or learning rate
        'clip': 0.1,  # Threshold to clip the ratio during SGA
        'render': False,  # Render the human readable environment during rollout?
        'render_every_i': 100,  # how often to render the environment
        'agents': 5, # number of meta-agents per env
        'epochs': 1000, # number of outer loop epochs (half of EPG implementation)
        'V': 64 # number of noise vectors
    }

    if args.mode == 'meta_train':  # most of the meta-training logic is in this function

        # gym env must have both continuous observation and action spaces.
        envs = list(gym.make('Pendulum-v1'), gym.make('MountainCarContinuous-v0'))
        # env = gym.make('LunarLanderContinuous-v2')

        manager = Manager()
        # dict of final mean returns from each agent
        mean_returns = manager.dict()
        processes = []

        loss_fn = LossNN(2, 1, mode="training")
        # loop through both environments
        for env in envs:
            # create outer loop for loss update
            for epoch in hyperparameters.epochs:
                vectors = list(np.random.multivariate_normal(
                    0, 1) * hyperparameters.V)

                # create inner loop each agent
                for agent in hyperparameters.agents:
                    p = Process(target=meta_train, args=(
                        env[agent], agent, loss_fn, hyperparameters, mean_returns))
                    processes.append(p)
                    p.start()
                for process in processes:
                    process.join()
                agg_return = np.sum(mean_returns) / hyperparameters.agents

                # update loss function parameters using agg_return and random vector pertubation
                loss_fn = loss_fn + 1/hyperparameters.V * \
                    np.sum(vectors) * (agg_return /
                                       (hyperparameters.agents/hyperparameters.V)) * random.choice(vectors)
    else:
        # single environment only for meta-testing
        test_env = gym.make('Pendulum-v1')

        meta_test(env=test_env, hyperparameters=hyperparameters,
                  loss_fn=args.loss_fn)


if __name__ == '__main__':
    args = get_args()  # Parse arguments from command line
    main(args)
