from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt 
from collections import deque


def main():
    env = 'Pendulum'
    mypath = f"./score_logs/{env}"
    fileNames = [f for f in listdir(mypath) if isfile(join(mypath, f))] 
    all_scores = []

    for filename in fileNames:
        print(filename)
        fullpath = mypath + "/" + filename
        file = np.loadtxt(fullpath)
        
        latest_scores = deque(maxlen=100) # store the latest 100 scores
        average_latest_scores = []

        for row in file:
            latest_scores.append(row)
            average_latest_scores.append(np.mean(latest_scores))

        all_scores.append(average_latest_scores)
    
    colors = ["b","g","r"]
    min_list_len = min([len(lst) for lst in all_scores])
    x = np.arange(min_list_len)
    ax = plt.subplot(111)
    for i in range(len(all_scores)):
        shortened_list = all_scores[i][:min_list_len]
        ax.plot(x, shortened_list, label=fileNames[i][:-4], color=colors[i], linewidth=0.8, markersize=2)

    ax.legend()

    plt.xlabel("Episode")
    plt.ylabel("Last 100 average scores")
    plt.title(f"Comparison of training performance of each algorithm in {env}")
    plt.savefig(f"plots/combined-results/{env}-combined-results.png")

main()