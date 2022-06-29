import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file, window=100):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-window):(i+1)])
    plt.ylabel('Score')       
    plt.xlabel('Episode')   
    plt.plot(x, running_avg)
    plt.title(f'Running average of previous {window} scores')
    plt.savefig(figure_file)
