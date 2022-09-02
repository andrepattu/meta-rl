from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt 
from collections import deque


def main():
    env = 'MountainCar'
    mypath = f"./score_logs/{env}"
    fileNames = [f for f in listdir(mypath) if isfile(join(mypath, f))] 
    all_scores = []

    for filename in fileNames:
        fullpath = mypath + "/" + filename
        file = np.loadtxt(fullpath)
        
        latest_scores = deque(maxlen=100) # store the latest 100 scores
        average_latest_scores = []

        for row in file:
            latest_scores.append(row)
            average_latest_scores.append(np.mean(latest_scores))

        all_scores.append(average_latest_scores)
    
    min_list_len = min([len(lst) for lst in all_scores])


    # # do the same for meta-testing scores
    # mypath2 = mypath + '/meta-testing'
    # fileNames2 = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]
    # all_scores2 = dict()
    # for idx, filename2 in enumerate(fileNames2):
    #     fullpath = mypath2 + "/" + filename2
    #     file = np.loadtxt(fullpath)
        
    #     latest_scores = deque(maxlen=100) # store the latest 100 scores
    #     average_latest_scores = []

    #     for row in file:
    #         latest_scores.append(row)
    #         average_latest_scores.append(np.mean(latest_scores))

    #     all_scores2[idx] = average_latest_scores[:min_list_len]

    colors = ["b","g","r","c","m"]
    x = np.arange(min_list_len)
    ax = plt.subplot(111)
    for i in range(len(all_scores)):
        shortened_list = all_scores[i][:min_list_len]
        ax.plot(x, shortened_list, label=fileNames[i][:-4], color=colors[i], linewidth=0.8, markersize=2)

    # ax.fill_between(x, all_scores2[0],all_scores2[1],all_scores2[2],all_scores2[3],all_scores2[4])
    # ax.plot(x, (all_scores2[0]+all_scores2[1]+all_scores2[2]+all_scores2[3]+all_scores2[4])/5, label=f'meta-ppo-{env}-testing', color='y', linewidth=0.8, markersize=2)

    ax.legend()

    plt.xlabel("Episode")
    plt.ylabel("Last 100 average scores")
    plt.title(f"Comparison of training/meta-testing performances of each algorithm in {env}")
    plt.savefig(f"plots/combined-results/{env}-combined-results.png")

main()