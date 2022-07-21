
def _log_summary(ep_length, ep_return, ep_num):
		"""
			Print the logs for the most recent episode.
		"""
		ep_length = str(round(ep_length, 2))
		ep_return = str(round(ep_return, 2))

		print(f"-------------------- Episode #{ep_num} --------------------")
		print(f"Episodic Length: {ep_length}")
		print(f"Episodic Return: {ep_return}")
		print(f"------------------------------------------------------")

def rollout(policy, env, render):
	"""
		Returns a generator to roll out each episode

		Parameters:
			policy - trained actor model to test
			env - environment to evaluate the policy on
			render - flag for rendering environment, default is false
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.
	"""
	# tests for 10 iterations
	for _ in range(10):
		done = False
		timesteps = 0
		ep_length = 0            
		ep_return = 0  
		obs = env.reset()
		          
		while not done:
			# Render environment if flag is true, default is false
			if render:
				env.render()

			# Query deterministic action from policy and step in the environment
			action = policy(obs).detach().numpy()
			obs, reward, done, _ = env.step(action)

			# Accumlate episodic rewards accross all timesteps
			ep_return += reward

			timesteps += 1
			
		# Track episodic length
		ep_length = timesteps

		# Return episodic length and episodic return
		yield ep_length, ep_return

def eval_policy(policy, env, render=False):
	"""
		Iterate the rollout generator object, which simulates each episode and return the most recent episode's length and return.

		Parameters:
			policy - trained actor model
			env -  environment to test the policy on
			render - flag for rendering environment, default is false

		Return:
			None
	"""
	total_score = 0

	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_length, ep_return) in enumerate(rollout(policy, env, render)):
		_log_summary(ep_length=ep_length, ep_return=ep_return, ep_num=ep_num)
		total_score += ep_return

	avg_score = total_score / 10
	print(f"average score over 10 iterations: {avg_score}")