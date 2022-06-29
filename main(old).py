import gym
import numpy as np
import time
from ppo import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    # env = gym.make('CartPole-v1') # action space is Discrete(2), observation shape is Box(4,)

    env = gym.make('ALE/Breakout-ram-v5') # action space is Discrete(4), observation shape is Box(128,)
    # env = gym.make('ALE/Boxing-ram-v5') # action space is Discrete(18), observation shape is Box(128,)

    num_eps = 1000 # 300

    mem_len = 20 # memory length
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)

    # figure_file = 'plots/cartpole.png'
    figure_file = 'plots/breakout.png'
    # figure_file = 'plots/boxing.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    t0 = time.time()

    for i in range(num_eps):
        observation = env.reset() 
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % mem_len == 0:
                agent.learn()
                learn_iters += 1
            observation = next_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # if avg_score > best_score:
        #     best_score = avg_score
        #     agent.save_models()

        if i % 10 == 0:
            x = [i+1 for i in range(len(score_history))]
            print(f'episode: {i}, score: {score:.1f}, avg score: {avg_score:.1f}, time_steps: {n_steps}, learning_steps: {learn_iters}, Time Elapsed: {time.time()-t0:.2f}')
            plot_learning_curve(x, score_history, figure_file)
