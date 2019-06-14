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
import Plot
import time
from statistics import mean, stdev
import ExpectedValue
import Plot_Histograms


cd = os.getcwd()
path = "./../all_results/Correct (Pickled)/Gamma99/output/Pickles/"
resPath = "./../Histograms/"
pickle_path = "./../Pickles/"
os.chdir(path)


for file in tqdm(glob.glob("*.pkl")):
    with open(str(file), 'rb') as f:
        agents_to_plot_RIGHT, agents_to_plot_LEFT, playerinfo, playerlearning, allAvgEU, error_bar, expUtility_LEFT, expUtility_RIGHT, allVals = dill.load(f)
    BOT_avg_expV = []
    MID_avg_expV = []
    TOP_avg_expV = []
    std = [[], [], [], [], [], []]
    outliers_std = [[], [], []]
    error_bar = []
    expUtility_LEFT = {}
    expUtility_RIGHT = {}
    # allVals -> (200,7) , 200 rounds, 7 values each round
    # 7 values are: simulation, allAvgEU, players_left, players_right, 1, std, outliers_std
    for i in range(len(allVals)):
        simulation = allVals[i][0]
        players_left = allVals[i][2]
        players_right = allVals[i][3]
        BOT_avg_expV, MID_avg_expV, TOP_avg_expV, std, outliers_std, expUtility_LEFT, expUtility_RIGHT = ExpectedValue.getExpVal(simulation, BOT_avg_expV, MID_avg_expV, TOP_avg_expV, players_left, players_right, 1, std, outliers_std)
    error_bar.append(stdev(std[0]) / (len(std[0]) ** 0.5))
    error_bar.append(stdev(std[1]) / (len(std[1]) ** 0.5))
    error_bar.append(stdev(std[2]) / (len(std[2]) ** 0.5))
    error_bar.append(stdev(std[3]) / (len(std[3]) ** 0.5))
    error_bar.append(stdev(std[4]) / (len(std[3]) ** 0.5))
    error_bar.append(stdev(std[5]) / (len(std[3]) ** 0.5))

    error_bar = [round(x, 2) for x in error_bar]
    BOT_avg_expV = [round(x, 2) for x in BOT_avg_expV]
    MID_avg_expV = [round(x, 2) for x in MID_avg_expV]
    TOP_avg_expV[0] = round(TOP_avg_expV[0], 2)
    TOP_avg_expV[1] = round(TOP_avg_expV[1], 2)

    simName = file[file.rfind('|') + 2:-4]

    with open("./../results.txt", "a") as myfile:
        myfile.write("\n\n" + simName + ": \n" + "BOTTOM expV: \t" + str(BOT_avg_expV) + "\nMID expV: \t" + str(MID_avg_expV) + "\nTOP expV: \t" + str(TOP_avg_expV) + " \n± " + str(error_bar) + "\n_______________________________________\n")

    simName = simName + "pkl|"
    os.chdir(resPath)
    hist_where = os.getcwd()
    Plot_Histograms.plotAgents(expUtility_LEFT, hist_where+"/", simName, "Left", agents_to_plot_LEFT[0].gamma)
    Plot_Histograms.plotAgents(expUtility_RIGHT, hist_where+"/", simName, "Left", agents_to_plot_RIGHT[0].gamma)
    os.chdir(pickle_path)

