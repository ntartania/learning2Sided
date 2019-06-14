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

# [knn_knn , knn_Exploration, Exploration_knn, Exploration_Exploration]
BOT_toplot = [[], [], [], []]
MID_toplot = [[], [], [], []]
TOP_toplot = [[], [], [], []]
cd = os.getcwd()
path = "./../all_results/PlotGammas/Pickles/"
res_Path = './../'
os.chdir(path)
all_pickles = [
    "1gamma70 KNN vs KNN.pkl",
    "2gamma70 KNN vs QLearning2 EXPLORATION.pkl",
    "3gamma70 QLearning2 EXPLORATION vs KNN.pkl",
    "4gamma70 QLearning2 EXPLORATION vs QLearning2 EXPLORATION.pkl",
    "1gamma75 KNN vs KNN.pkl",
    "2gamma75 KNN vs QLearning2 EXPLORATION.pkl",
    "3gamma75 QLearning2 EXPLORATION vs KNN.pkl",
    "4gamma75 QLearning2 EXPLORATION vs QLearning2 EXPLORATION.pkl",
    "1gamma80 KNN vs KNN.pkl",
    "2gamma80 KNN vs QLearning2 EXPLORATION.pkl",
    "3gamma80 QLearning2 EXPLORATION vs KNN.pkl",
    "4gamma80 QLearning2 EXPLORATION vs QLearning2 EXPLORATION.pkl",
    "1gamma85 KNN vs KNN.pkl",
    "2gamma85 KNN vs QLearning2 EXPLORATION.pkl",
    "3gamma85 QLearning2 EXPLORATION vs KNN.pkl",
    "4gamma85 QLearning2 EXPLORATION vs QLearning2 EXPLORATION.pkl",
    "1gamma90 KNN vs KNN.pkl",
    "2gamma90 KNN vs QLearning2 EXPLORATION.pkl",
    "3gamma90 QLearning2 EXPLORATION vs KNN.pkl",
    "4gamma90 QLearning2 EXPLORATION vs QLearning2 EXPLORATION.pkl",
    "1gamma95 KNN vs KNN.pkl",
    "2gamma95 KNN vs QLearning2 EXPLORATION.pkl",
    "3gamma95 QLearning2 EXPLORATION vs KNN.pkl",
    "4gamma95 QLearning2 EXPLORATION vs QLearning2 EXPLORATION.pkl",
    "1gamma99 KNN vs KNN.pkl",
    "2gamma99 KNN vs QLearning2 EXPLORATION.pkl",
    "3gamma99 QLearning2 EXPLORATION vs KNN.pkl",
    "4gamma99 QLearning2 EXPLORATION vs QLearning2 EXPLORATION.pkl",
    ]
#
# for file in tqdm(all_pickles):
#     try:
#         with open(str(file), 'rb') as f:
#             agents_RIGHT, agents_LEFT, playerinfo, playerlearning, BOT_avg_expV, MID_avg_expV, TOP_avg_expV, error_bar, expUtility_LEFT, expUtility_RIGHT, allVals = dill.load(f)
#     except:
#         with open(str(file), 'rb') as f:
#             agents_RIGHT, agents_LEFT, playerinfo, playerlearning, ALLavg_expV, error_bar, expUtility_LEFT, expUtility_RIGHT, allVals = dill.load(f)
#         BOT_avg_expV = []
#         MID_avg_expV = []
#         TOP_avg_expV = []
#         std = [[], [], [], [], [], []]
#         outliers_std = [[], [], []]
#         error_bar = []
#         expUtility_LEFT = {}
#         expUtility_RIGHT = {}
#         # allVals -> (200,7) , 200 rounds, 7 values each round
#         # 7 values are: simulation, allAvgEU, players_left, players_right, 1, std, outliers_std
#         for i in range(len(allVals)):
#             simulation = allVals[i][0]
#             players_left = allVals[i][2]
#             players_right = allVals[i][3]
#             BOT_avg_expV, MID_avg_expV, TOP_avg_expV, std, outliers_std, expUtility_LEFT, expUtility_RIGHT = ExpectedValue.getExpVal(simulation, BOT_avg_expV, MID_avg_expV, TOP_avg_expV, players_left, players_right, 1, std, outliers_std)
#         error_bar.append(stdev(std[0]) / (len(std[0]) ** 0.5))
#         error_bar.append(stdev(std[1]) / (len(std[1]) ** 0.5))
#         error_bar.append(stdev(std[2]) / (len(std[2]) ** 0.5))
#         error_bar.append(stdev(std[3]) / (len(std[3]) ** 0.5))
#         error_bar.append(stdev(std[4]) / (len(std[3]) ** 0.5))
#         error_bar.append(stdev(std[5]) / (len(std[3]) ** 0.5))
#
#         error_bar = [round(x, 2) for x in error_bar]
#         BOT_avg_expV = [round(x, 2) for x in BOT_avg_expV]
#         MID_avg_expV = [round(x, 2) for x in MID_avg_expV]
#         TOP_avg_expV[0] = round(TOP_avg_expV[0], 2)
#         TOP_avg_expV[1] = round(TOP_avg_expV[1], 2)
#
#     l = 0
#     r = 1
#     if file[0] == '1':
#         BOT_toplot[0].append(BOT_avg_expV[l])
#         MID_toplot[0].append(MID_avg_expV[r])
#         TOP_toplot[0].append(TOP_avg_expV[l])
#     elif file[0] == '2':
#         if file[1:8] == 'gamma95':
#             r = l
#         BOT_toplot[1].append(BOT_avg_expV[l])
#         MID_toplot[1].append(MID_avg_expV[r])
#         TOP_toplot[1].append(TOP_avg_expV[l])
#     elif file[0] == '3':
#         if file[1:8] == 'gamma95':
#             l = r
#         BOT_toplot[2].append(BOT_avg_expV[r])
#         MID_toplot[2].append(MID_avg_expV[l])
#         TOP_toplot[2].append(TOP_avg_expV[r])
#     elif file[0] == '4':
#         BOT_toplot[3].append(BOT_avg_expV[l])
#         MID_toplot[3].append(MID_avg_expV[r])
#         TOP_toplot[3].append(TOP_avg_expV[l])
#
# with open(res_Path+"valuesForGAMMAS.pkl", "wb") as dill_file:
#     dill.dump([BOT_toplot, MID_toplot, TOP_toplot], dill_file)
#


with open(res_Path+"valuesForGAMMAS.pkl", 'rb') as f:
    BOT_toplot, MID_toplot, TOP_toplot = dill.load(f)

x = [1,2,3,4,5,6,7]
myTicks = ['0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '0.99']
fig, ax= plt.subplots()
ax.plot(x,BOT_toplot[0], marker='h', linestyle = '--')
ax.plot(x,BOT_toplot[1], marker='^', linestyle = 'solid')
ax.plot(x,BOT_toplot[2], marker='s', linestyle = '-.')
ax.plot(x,BOT_toplot[3], marker='*', linestyle = ':')
ax.legend(['Modellers vs Modellers', 'Modellers vs Qlearners', 'Qlearners vs Modellers', 'Qlearners vs Qlearners'], fontsize=12)
ax.set_xlabel("Gamma Values", fontsize=15)
ax.set_ylabel("Expected Utility", fontsize=15)
ax.set_title("Bottom Agents [1 to 6]", fontsize=19)
ax.set_xticks(x)
ax.set_xticklabels(myTicks)
fig.savefig(res_Path+"BottomGAMMAS.pdf")
plt.close(fig)


fig2, ax2= plt.subplots()
ax2.plot(x,MID_toplot[0], marker='h', linestyle = '--')
ax2.plot(x,MID_toplot[1], marker='^', linestyle = 'solid')
ax2.plot(x,MID_toplot[2], marker='s', linestyle = '-.')
ax2.plot(x,MID_toplot[3], marker='*', linestyle = ':')
ax2.legend(['Modellers vs Modellers', 'Modellers vs Qlearners', 'Qlearners vs Modellers', 'Qlearners vs Qlearners'], fontsize=12)
ax2.set_xlabel("Gamma Values", fontsize=15)
ax2.set_ylabel("Expected Utility", fontsize=15)
ax2.set_title("Middle Agents [7 to 14]", fontsize=19)
ax2.set_xticks(x)
ax2.set_xticklabels(myTicks)
fig2.savefig(res_Path+"MiddleGAMMAS.pdf")
plt.close(fig2)

fig3, ax3= plt.subplots()
ax3.plot(x,TOP_toplot[0], marker='h', linestyle = '--')
ax3.plot(x,TOP_toplot[1], marker='^', linestyle = 'solid')
ax3.plot(x,TOP_toplot[2], marker='s', linestyle = '-.')
ax3.plot(x,TOP_toplot[3], marker='*', linestyle = ':')
ax3.legend(['Modellers vs Modellers', 'Modellers vs Qlearners', 'Qlearners vs Modellers', 'Qlearners vs Qlearners'], fontsize=12)
ax3.set_xlabel("Gamma Values", fontsize=14)
ax3.set_ylabel("Expected Utility", fontsize=14)
ax3.set_title("Top Agents [15 to 20]", fontsize=16)
ax3.set_xticks(x)
ax3.set_xticklabels(myTicks)
fig3.savefig(res_Path+"TopGAMMAS.pdf")
plt.close(fig3)


