import numpy as np
import Create_Agents
import random
import config
import Agent
import matplotlib.pyplot as plt
import Plot
import time
import ExpectedValue
import Plot_Histograms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dill
import glob
import os
from tqdm import tqdm
from statistics import mean, stdev

cd = os.getcwd()
path = "./../all_results/TheoHists/"
os.chdir(path)
gamma = 1

fig, ax = plt.subplots()
x = [1  , 2  , 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# y = [0.5, 1.8, 2,2,2,2,2,2,4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
y=x
ax.bar(x, y, width=1, color='b', align='center', alpha=0.8)
ax.plot(x, y, color='red', linestyle='dashed')
ax.set_xlabel("Agent Type", fontsize=15)
ax.set_ylabel("Expected Utility", fontsize=15)
ax.set_title("Theoretical vs Theoretical "+"\t"+r'$\gamma$:1', fontsize=19)#+str(gamma), fontsize=19)
ax.set_xticks(np.arange(1, 21, 1.0))
ax.set_yticks(np.arange(1, 21, 1.0))
fig.savefig("Hist_Theo_"+str(gamma)+".pdf")
plt.close(fig)