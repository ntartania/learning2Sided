import numpy as np
import Create_Agents
import random
import config
import Agent
import matplotlib.pyplot as plt
import dill
import glob
import os
from tqdm import tqdm


def plotAgents(myDict, resPath, pickle_name, side, gamma):
    fig, ax = plt.subplots()
    tup = sorted(myDict.items())
    x = [t[0] for t in tup]
    y = [t[1] for t in tup]
    ax.bar(myDict.keys(), myDict.values(), width=1, color='b', align='center', alpha=0.8)
    ax.plot(x, y, color='red', linestyle='dashed')
    ax.set_xlabel("Agent Type")
    ax.set_ylabel("Expected Utility")
    ax.set_xticks(np.arange(1, 21, 1.0))
    ax.set_yticks(np.arange(1, 21, 1.0))
    # t_from = pickle_name.rfind('|')+2
    t_end = pickle_name.rfind('|')
    left = pickle_name[:pickle_name.rfind('vs')-1]
    right = pickle_name[pickle_name.rfind('vs')+3:t_end]
    if side == "Left":
        ax.set_title(r"$\bf{"+left+"}$" + " vs " + right +"\t"+ r'$\gamma$:'+str(gamma))
    else:
        ax.set_title(left+ " vs " +r"$\bf{"+right+"}$" +"\t"+ r'$\gamma$:'+str(gamma))
    fig.savefig(resPath+pickle_name[:-4]+ " | " + side+".pdf")
    plt.close(fig)

def makeHist():
    # path = "/Users/Riccardo/Desktop/Edinburgh/MSc_Dissertation/Code/old_results/3_17_20_Gamma90/output/Pickles/"
    # resPath = "/Users/Riccardo/Desktop/Edinburgh/MSc_Dissertation/Code/old_results/3_17_20_Gamma90/output/Histograms/"
    path = "output/Pickles/"
    resPath = "../Histograms/"
    cd=os.getcwd()
    os.chdir(path)

    # for file in tqdm(glob.glob("*.pkl")):
    for file in glob.glob("*.pkl"):
        with open(str(file), 'rb') as f:
            agents_to_plot_RIGHT, agents_to_plot_LEFT, playerinfo, playerlearning, BOT_avg_expV, MID_avg_expV, TOP_avg_expV, error_bar, expUtility_LEFT, expUtility_RIGHT, allVals = dill.load(f)
        plotAgents(expUtility_LEFT, resPath, file, "Left", agents_to_plot_LEFT[0].gamma)
        plotAgents(expUtility_RIGHT, resPath, file, "Right", agents_to_plot_RIGHT[0].gamma)
