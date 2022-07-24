import sys
import argparse
import gym
import gym.wrappers as wrap
import numpy as np
import torch

from ddpg import DDPG
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
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = DDPG(alph=hyperparameters['alph'], beta=hyperparameters['beta'], in_dim=[obs_dim], tau=hyperparameters['tau'], 
            env_name=env_name, batch_size=hyperparameters['batch_size'],  layer1_size=hyperparameters['l1_size'], 
            layer2_size=hyperparameters['l2_size'], act_dim=act_dim)

    # Loads in an existing actor/critic model to resume training if specified
    if actor_model != '' or critic_model != '': # Prevents rewriting of a model pth file if I did not specify both models
        print(f"Please specify both actor and critic models!")
        sys.exit(0)
    elif actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...")
        agent.actor.load_state_dict(torch.load(actor_model))
        agent.critic.load_state_dict(torch.load(critic_model))
    else:
        print(f"Training from scratch.")

    score_history = []
    best_score = env.reward_range[0]
    for i in range(hyperparameters['timesteps']):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.query_action(obs)
            next_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, next_state, int(done))
            agent.learn()
            score += reward
            obs = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        if i % 10 == 0:
            print(f'episode: {i}, score: {score:.2f}, trailing 100 games avg: {np.mean(score_history[-100:]):.2f}')

            x = [i+1 for i in range(len(score_history))]
            filename = f'plots/{env_name}_test.png'
            plot_learning_curve(x, score_history, filename, window=100)

def main(args):
	"""
		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# THESE HYPERPARAMETERS ARE THE ONES THAT PRODUCED THE CURRENT BASELINE
	hyperparameters = {
				'alph': 0.000025, 
				'beta': 0.00025, 
				'tau': 0.001, 
				'batch_size': 64,
				'l1_size': 400,
				'l2_size': 300,
                'timesteps': 1000,
			  }

	# gym env must have both continuous observation and action spaces.
	env_name = 'LunarLanderContinuous-v2'
    # env_name = 'Pendulum-v1'
    # env_name = 'MountainCarContinuous-v0'

	if args.mode == 'train':
		train(env_name=env_name, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
	else:
		test_env = wrap.ResizeObservation(env, shape=(2,1)) # pendulum (3,1), MountainCar obs_dim (2,1)
		# LunarLander does not work because of different actions space
		test(env=test_env, actor_model=args.actor_model)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)