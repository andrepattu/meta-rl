import sys
import argparse
import gym
import gym.wrappers as wrap
import torch

from ppo import PPO, FeedForwardNN
from eval_policy import eval_policy


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

    # can be 'train' or 'test'
    parser.add_argument('--mode', dest='mode', type=str, default='train')
    parser.add_argument('--actor_model', dest='actor_model',
                        type=str, default='')     # your actor model filename
    parser.add_argument('--critic_model', dest='critic_model',
                        type=str, default='')   # your critic model filename

    args = parser.parse_args()

    return args


def train(env, hyperparameters, actor_model, critic_model):
    """
            Parameters:
                    env - the environment to train on
                    hyperparameters - a dict of hyperparameters to use, defined in main
                    actor_model - the actor model to load in for training
                    critic_model - the critic model to load in for training

            Return:
                    None
    """
    print("Training")
    model = PPO(policy=FeedForwardNN, env=env, **hyperparameters)

    # Loads in an existing actor/critic model to resume training if specified
    # Prevents rewriting of a model pth file if I did not specify both models
    if actor_model != '' or critic_model != '':
        print(f"Please specify both actor and critic models!")
        sys.exit(0)
    elif actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...")
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
    else:
        print(f"Training from scratch.")

    # total_timesteps is set high but can be killed if learning has converged
    model.train(total_timesteps=100_000_000)


def test(env, actor_model):
    """
            Parameters:
                    env - the environment to test the policy on
                    actor_model - the actor model to load in to test

            Return:
                    None
    """
    print(f"Testing {actor_model}")

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.")
        sys.exit(0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = FeedForwardNN(obs_dim, act_dim, mode="testing")
    # strict is set to False here to catch exceptions where the networks don't exactly match up
    policy.load_state_dict(torch.load(actor_model), strict=False)

    # seperate function to evalaute trained policy
    eval_policy(policy=policy, env=env, render=True)


def main(args):
    """
            Parameters:
                    args - the arguments parsed from command line

            Return:
                    None
    """
    # THESE HYPERPARAMETERS ARE THE ONES THAT PRODUCED THE ORIGINAL/META-PPO BASELINE
    hyperparameters = {
        'timesteps_per_batch': 2048,
        'max_timesteps_per_episode': 200,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 3e-4,
        'clip': 0.2,
        'render': False,  # True
        'render_every_i': 10
    }

    # # THESE HYPERPARAMETERS ARE FOR THE WEAK/PPO BASELINE
    # hyperparameters = {
    # 			'timesteps_per_batch': 4096, # Number of timesteps to run per batch
    # 			'timesteps_per_episode': 256, # Number of timesteps per episode
    # 			'gamma': 0.95, # Discount factor to be applied when calculating Rewards-To-Go
    # 			'num_epochs': 15, # Number of epochs to update actor/critic per iteration
    # 			'alph': 3e-3, # alpha or learning rate
    # 			'clip': 0.1, # Threshold to clip the ratio during SGA
    # 			'render': False, # Render the human readable environment during rollout?
    # 			'render_every_i': 100 # how often to render the environment
    # 		  }

    # gym env must have both continuous observation and action spaces.
    env = gym.make('Pendulum-v1')
    # env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('LunarLanderContinuous-v2')

    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters,
              actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        # pendulum (3,1), MountainCar obs_dim (2,1)
        test_env = wrap.ResizeObservation(env, shape=(2, 1))
        # LunarLander does not work because of different actions space
        test(env=test_env, actor_model=args.actor_model)


if __name__ == '__main__':
    args = get_args()  # Parse arguments from command line
    main(args)
