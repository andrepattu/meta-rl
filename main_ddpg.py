import sys
import os
import argparse
import gym
import gym.wrappers as wrap
import numpy as np
import torch

from ddpg import DDPG, Network
from eval_policy_ddpg import eval_policy
from utils import plot_learning_curve

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

	parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # your actor model filename
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # your critic model filename

	args = parser.parse_args()

	return args


def train(env_name, hyperparameters, actor_model, critic_model):
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

    env = gym.make(env_name)
    custom_name = 'ddpg_' + env_name + '_old_baseline' # custom name for experiment
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = DDPG(alph=hyperparameters['alph'], beta=hyperparameters['beta'], in_dim=[obs_dim], act_dim=act_dim, custom_name=custom_name,
            tau=hyperparameters['tau'], batch_size=hyperparameters['batch_size'],  l1_size=hyperparameters['l1_size'], 
            l2_size=hyperparameters['l2_size'], gamma=hyperparameters['gamma'], replay_buffer_size=hyperparameters['replay_buffer_size'])

    score_history = []
    best_score = env.reward_range[0]

    path_to_logs = f"score_logs/{custom_name}.txt"
    # remove existing txt file so that scores are not appended to old logs
    try:
        os.remove(path_to_logs)
    except OSError:
        pass

    for i in range(hyperparameters['timesteps']):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.query_action(obs)
            next_state, reward, done, info = env.step(act)
            # print(info) # for debugging
            agent.store_memory(obs, act, reward, next_state, int(done))
            agent.learn()
            score += reward
            obs = next_state
        score = round(score, 2) # round score to 2 dp
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # write scores to score logs
        with open(path_to_logs, 'a') as f:
            f.write(str(score))
            f.write('\n')

        if avg_score > best_score: #save models when avg score has improved
            best_score = avg_score
            agent.save_models()

        if i % 10 == 0:
            print(f'episode: {i}, score: {score:.2f}, trailing 100 games avg: {np.mean(score_history[-100:]):.2f}')

            x = [i+1 for i in range(len(score_history))]
            filename = f'plots/{custom_name}.png'
            plot_learning_curve(x, score_history, filename, window=100)

def test(env_name, hyperparameters, actor_model):
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

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = Network(lr=hyperparameters['alph'], in_dim=[obs_dim], act_dim=act_dim, l1_size=hyperparameters['l1_size'], 
                    l2_size=hyperparameters['l2_size'], name=env_name+'Actor')
    # strict is set to False here to catch exceptions where the networks don't exactly match up
    policy.load_state_dict(torch.load(actor_model), strict=False) 

    # agent = DDPG(alph=hyperparameters['alph'], beta=hyperparameters['beta'], in_dim=[obs_dim], act_dim=act_dim, env_name=env_name,
    #         tau=hyperparameters['tau'], batch_size=hyperparameters['batch_size'],  l1_size=hyperparameters['l1_size'], 
    #         l2_size=hyperparameters['l2_size'], gamma=hyperparameters['gamma'], replay_buffer_size=hyperparameters['replay_buffer_size'])
    # agent.load_state_dict(torch.load(actor_model), strict=False) # strict is set to False here to catch exceptions where the networks don't exactly match up

    # seperate function to evaluate trained policy
    eval_policy(policy=policy, env=env, render=True)

def main(args):
	"""
		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# THESE HYPERPARAMETERS ARE THE ONES THAT PRODUCED THE OLD BASELINE
	hyperparameters = {
				'alph': 0.000025, 
				'beta': 0.00025, 
				'tau': 0.001, 
				'batch_size': 64,
				'l1_size': 400,
				'l2_size': 300,
                'gamma': 0.99,
                'replay_buffer_size': 1000000,
                'timesteps': 1000,
			  }

	# # THESE HYPERPARAMETERS ARE THE ONES THAT SHOULD PRODUCE A NEW BASELINE
	# hyperparameters = {
	# 			'alph': 0.0001, # actor's learning rate
	# 			'beta': 0.001, # critic's learning rate
	# 			'tau': 0.002, # target update factor
	# 			'batch_size': 32,
	# 			'l1_size': 256,
	# 			'l2_size': 128,
    #             'gamma': 0.98,
    #             'replay_buffer_size': 100000,
    #             'timesteps': 10000,
	# 		  }

	# gym env must have both continuous observation and action spaces.
	env_name = 'LunarLanderContinuous-v2'
    # env_name = 'Pendulum-v1'
    # env_name = 'MountainCarContinuous-v0'

	if args.mode == 'train':
		train(env_name=env_name, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
	else:
		# test_env = wrap.ResizeObservation(env, shape=(2,1)) # pendulum (3,1), MountainCar obs_dim (2,1)
		# LunarLander does not work because of different actions space
		test(env_name=env_name, hyperparameters=hyperparameters, actor_model=args.actor_model)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)