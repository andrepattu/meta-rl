import gym
import numpy as np

from ddpg import DDPG
from utils import plot_learning_curve

env_name = 'LunarLanderContinuous-v2'
# env_name = 'Pendulum-v1'
# env_name = 'MountainCarContinuous-v0'

env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
agent = DDPG(alph=0.000025, beta=0.00025, in_dim=[obs_dim], tau=0.001, env_name=env_name,
            batch_size=64,  layer1_size=400, layer2_size=300, act_dim=act_dim)

score_history = []
best_score = env.reward_range[0]
for i in range(1000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        next_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, next_state, int(done))
        agent.learn()
        score += reward
        obs = next_state
        #env.render()
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