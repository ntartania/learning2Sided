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
path = "./../all_results/matchingSet/theo/"
os.chdir(path)

pickleFile = "Theoretical_Theoretical_99.pkl"

try:
    with open(pickleFile, 'rb') as f:
        agents_RIGHT, agents_LEFT, playerinfo, playerlearning, BOT_avg_expV, MID_avg_expV, TOP_avg_expV, error_bar, expUtility_LEFT, expUtility_RIGHT, allVals = dill.load(f)
except:
    with open(pickleFile, 'rb') as f:
        agents_RIGHT, agents_LEFT, playerinfo, playerlearning, ALLavg_expV, error_bar, expUtility_LEFT, expUtility_RIGHT, allVals = dill.load(f)

x_RIGHT = np.array([])
y_RIGHT = np.array([])

for i in agents_RIGHT:
    gamma = i.gamma
    i.utility = [x for x in i.utility]# if x != 0]
    x_R_temp = np.full((len(i.utility)), i.get_type()[0])
    x_RIGHT = np.concatenate((x_RIGHT, x_R_temp))
    y_RIGHT = np.concatenate((y_RIGHT, i.utility))
print(x_RIGHT)
print(y_RIGHT)
exit(98)
# x_RIGHT = list(range(1, 21))
# y_RIGHT = list(range(1, 21))
# gamma = 1
fig, ax= plt.subplots()
ax.plot(x_RIGHT,y_RIGHT,'o')
ax.set_xlabel("Right Agent Type", fontsize=15)
ax.set_ylabel("Left Agent Type", fontsize=15)
ax.set_title("Matching Set " + r'   $\gamma$:'+str(gamma), fontsize=19)
ax.set_xticks(np.arange(min(x_RIGHT), max(x_RIGHT)+1, 1.0))
ax.set_yticks(np.arange(min(x_RIGHT), max(x_RIGHT)+1, 1.0))

fig.savefig("Matching_Set_"+str(gamma)+".pdf")
plt.close(fig)