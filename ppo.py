import gym
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal

from utils import plot_learning_curve

class FeedForwardNN(nn.Module):
	def __init__(self, in_dim, out_dim, mode):
		"""
			Initialize the network and set up the layers.
			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int
			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.mode = mode # training or testing?
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		# THESE HYPERPARAMETERS ARE THE ONES THAT PRODUCED THE CURRENT BASELINE
		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)
		
		# # THESE HYPERPARAMETERS ARE SUGGESTED AFTER REFACTORING TO IMPROVE THE BASELINE PERFORMANCE
		# self.layer1 = nn.Linear(in_dim, 128)
		# self.layer2 = nn.Linear(128, 128)
		# self.layer3 = nn.Linear(128, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.
			Parameters:
				obs - observation to pass as input
			Return:
				output - the output of the forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray) and self.mode == "testing": # with cpu
			obs = np.reshape(obs, -1) # flatten obs array before converting to tensor
			obs = torch.tensor(obs, dtype=torch.float)
		elif isinstance(obs, np.ndarray) and self.mode == "training": # with gpu
			obs = torch.tensor(obs, dtype=torch.float).to(self.device)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output

class PPO:
	def __init__(self, policy_class, env, **hyperparameters):
		"""
			Initializes the PPO agent with the defined hyperparameters.
			Parameters:
				policy_class - the policy class to use for the actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
			Returns:
				None
		"""
		# Make sure the input environment is compatible; continuous action and observation spaces
		assert(type(env.observation_space) == gym.spaces.Box)
		assert(type(env.action_space) == gym.spaces.Box)

		self._init_hyperparameters(hyperparameters)
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		# Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim, mode="training") # set mode flag to training by default                                                 
		self.critic = policy_class(self.obs_dim, 1, mode="training")  # set mode flag to training by default 

		# Set both networks to gpu device
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.actor.to(self.device)
		self.critic.to(self.device)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.alph)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.alph)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var).to(self.device)

		# Logs summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			'elapsed_timesteps': 0,  # elapsed timesteps
			'elapsed_iterations': 0, # iterations so far
			'batch_lengths': [],     # episodic lengths in batch
			'batch_returns': [],     # episodic returns in batch
			'actor_losses': [],      # actor losses in current iteration
		}

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks.
			Parameters:
				total_timesteps - the total number of timesteps to train for
			Return:
				None
		"""
		print(f"Learning... Running {self.timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		elapsed_timesteps = 0 
		elapsed_iterations = 0 

		self.logger['score_history'] = []
		
		while elapsed_timesteps < total_timesteps:                                                                      
			# Collecting the batch simulations (set of trajectories)
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lengths = self.rollout()                     

			self.logger['score_history'].append(self.logger['score'])

			# Calculate how many timesteps we collected this batch
			elapsed_timesteps += np.sum(batch_lengths)

			# Increment the number of iterations
			elapsed_iterations += 1

			# Logging elapsed timesteps and iterations
			self.logger['elapsed_timesteps'] = elapsed_timesteps
			self.logger['elapsed_iterations'] = elapsed_iterations

			# Compute advantage estimates at k-th iteration
			values, _ = self.evaluate(batch_obs, batch_acts)
			Adv_k = batch_rtgs - values.detach()                                                                       

			# Normalize advantages to decrease variance and improve convergence much more stable and faster.
			Adv_k = (Adv_k - Adv_k.mean()) / (Adv_k.std() + 1e-10)

			# Update policy for number of epochs
			for _ in range(self.num_epochs):                                                       
				# Calculate values_phi and pi_theta(a_t | s_t)
				values, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				ratios = torch.exp(curr_log_probs - batch_log_probs).to(self.device)

				# Calculate surrogate losses.
				surr1 = ratios * Adv_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * Adv_k

				# Calculate actor and critic losses.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(values, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor losses
				self.logger['actor_losses'].append(actor_loss.detach())

			# Print a summary of the training
			self._log_summary()

			# Save the models if it's time
			if elapsed_iterations % self.save_freq == 0:
				torch.save(self.actor.state_dict(), 'models/test_ppo_actor.pth')
				torch.save(self.critic.state_dict(), 'models/test_ppo_critic.pth')

				# plot graph of average episodic return
				x = [i+1 for i in range(len(self.logger['score_history']))] # x is the graph's x-axis value
				plot_learning_curve(x, self.logger['score_history'], "plots/test.png")

	def rollout(self):
		"""
			Collect the data from each simulated batch. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.
			Parameters:
				None
			Return:
				batch_obs - observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lengths - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_returns = []
		batch_rtgs = []
		batch_lengths = []
		ep_rewards = [] # rewards collected per episode
		t = 0 # keeps track of elapsed timesteps

		# Keep simulating until the agent has run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rewards = [] # reinitialize to be empty each batch

			# Reset the environment
			obs = self.env.reset() # observation
			done = False

			# Run an episode for the number of timesteps
			for ep in range(self.timesteps_per_episode):
				# If render is specified, render the environment
				if self.render and (self.logger['elapsed_iterations'] % self.render_every_i == 0) and len(batch_lengths) == 0:
					self.env.render()

				t += 1 # Increment timesteps ran this batch so far

				# Track observations in this batch
				batch_obs.append(obs)

				# Calculate action and take a step in the env. 
				action, log_prob = self.get_action(obs)
				obs, reward, done, _ = self.env.step(action)

				# Track recent reward, action, and action log probability
				ep_rewards.append(reward)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)

				score = np.sum(ep_rewards)
				score = round(score, 2)
				self.logger['score'] = score

				if done:
					break

			# Track episodic lengths and rewards
			batch_lengths.append(ep + 1)
			batch_returns.append(ep_rewards)

		# Reshape data as tensors in the shape specified in function description
		batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float, device=self.device)
		batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float, device=self.device)
		batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float, device=self.device)
		batch_rtgs = self.compute_rtgs(batch_returns)                                                              

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_returns'] = batch_returns
		self.logger['batch_lengths'] = batch_lengths

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lengths

	def compute_rtgs(self, batch_returns):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.
			Parameters:
				batch_returns - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		batch_rtgs = []

		# Iterate through each episode
		for ep_rewards in reversed(batch_returns):
			discounted_reward = 0

			# Iterate through all rewards in the episode. Iteration is backwards for smoother calculation of each discounted return 
			for reward in reversed(ep_rewards):
				discounted_reward = reward + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)

		return batch_rtgs

	def get_action(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.
			Parameters:
				obs - the observation at the current timestep
			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Query the actor network for a mean action
		mean = self.actor(obs)

		# Create a distribution with the mean action and std from the covariance matrix above.
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)

		# Return the sampled action and the log probability of that action in our distribution
		return torch.Tensor.cpu(action).detach().numpy(), torch.Tensor.cpu(log_prob).detach().numpy()

	def evaluate(self, batch_obs, batch_acts):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.
			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)
			Return:
				values - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value for each batch_obs. Shape of values should be same as batch_rtgs
		values = self.critic(batch_obs).squeeze().to(self.device)

		# Calculate the log probabilities of batch actions using most recent actor network.
		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		return values, log_probs

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters
			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.
			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.timesteps_per_episode = 1600           	# Number of timesteps per episode
		self.num_epochs = 5                				# Number of epochs to update actor/critic per iteration
		self.alph = 0.005                               # Alpha or learning rate
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = True                              # Render the human readable environment during rollout?
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.
			Parameters:
				None
			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		elapsed_timesteps = self.logger['elapsed_timesteps']
		elapsed_iterations = self.logger['elapsed_iterations']
		avg_ep_lens = np.mean(self.logger['batch_lengths'])
		avg_ep_rewards = np.mean([np.sum(ep_rewards) for ep_rewards in self.logger['batch_returns']])
		avg_actor_loss = np.mean([torch.Tensor.cpu(losses).float().mean() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rewards = str(round(avg_ep_rewards, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{elapsed_iterations} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rewards}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Elapsed Timesteps: {elapsed_timesteps}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lengths'] = []
		self.logger['batch_returns'] = []
		self.logger['actor_losses'] = []