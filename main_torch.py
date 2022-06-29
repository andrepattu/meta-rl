from ddpg_torch import Agent
import gym
import numpy as np
from utils import plot_learning_curve

# environment = 'LunarLanderContinuous-v2'
# environment = 'Pendulum-v1'
environment = 'MountainCarContinuous-v0'

env = gym.make(environment)

if environment == 'LunarLanderContinuous-v2':
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)
elif environment == 'Pendulum-v1':
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[3], tau=0.001, env=env,
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=1) 
elif environment == 'MountainCarContinuous-v0':
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[2], tau=0.001, env=env,
                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=1) 

#agent.load_models()
np.random.seed(0)

score_history = []
best_score = env.reward_range[0]
for i in range(1000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        #env.render()
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    if i % 10 == 0:
        print('episode ', i, 'score %.2f' % score, 'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

        x = [i+1 for i in range(len(score_history))]
        filename = f'plots/{environment}.png'
        plot_learning_curve(x, score_history, filename, window=100)