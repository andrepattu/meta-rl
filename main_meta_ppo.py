import argparse
import gym
import gym.wrappers as wrap
import sys
import torch

from meta_ppo import Meta_PPO, FeedForwardNN, LossNN
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

	parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # your actor model filename
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # your critic model filename

	args = parser.parse_args()

	return args

def train(env, hyperparameters, actor_model, critic_model):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in for training
			critic_model - the critic model to load in for training

		Return:
			None
	"""	
	print(f"Training", flush=True)

	# Create a model for PPO.
	model = Meta_PPO(policy_class=FeedForwardNN, loss_fn=LossNN, env=env, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print(f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	# kill the process whenever you feel like PPO is converging
	model.learn(total_timesteps=200_000_000)

def test(env, actor_model):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in to test

		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	# Build the policy network with correct dimensions
	policy = FeedForwardNN(obs_dim, act_dim, mode="testing")

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model), strict=False)

	# seperate module that evalautes the trained policy weights
	eval_policy(policy=policy, env=env, render=True)

def main(args):
	"""
		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# THESE HYPERPARAMETERS ARE THE ONES THAT PRODUCED THE CURRENT BASELINE
	hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': False, # True
				'render_every_i': 10
			  }

	# # THESE HYPERPARAMETERS ARE SUGGESTED AFTER REFACTORING TO IMPROVE THE BASELINE PERFORMANCE
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

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
	else:
		test_env = wrap.ResizeObservation(env, shape=(2,1)) # pendulum (3,1), MountainCar obs_dim (2,1)
		# LunarLander does not work because of different actions space
		test(env=test_env, actor_model=args.actor_model)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)
