import gym
import ray
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.worker_set import WorkerSet

# Setup policy and rollout workers.
environment_name1 = "Pendulum-v1"
env = gym.make(environment_name1)
policy = PPOTorchPolicy(env.observation_space, env.action_space, {})
workers = WorkerSet(
    policy_class=PPOTorchPolicy,
    env_creator=lambda c: gym.make(environment_name1),
    num_workers=8) #8

while True:
    # Gather a batch of samples.
    T1 = SampleBatch.concat_samples(
        ray.get([w.sample.remote() for w in workers.remote_workers()]))

    # Improve the policy using the T1 batch.
    policy.learn_on_batch(T1)

    # The local worker acts as a "parameter server" here.
    # We put the weights of its `policy` into the Ray object store once (`ray.put`)...
    weights = ray.put({"default_policy": policy.get_weights()})
    for w in workers.remote_workers():
        # ... so that we can broacast these weights to all rollout-workers once.
        w.set_weights.remote(weights)