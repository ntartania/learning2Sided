import numpy as np
import Create_Agents
import random
import config
import Agent
import matplotlib.pyplot as plt
import dill
from scipy import stats

def plotRes():
    path = "output/Pickles/"
    resPath = "output/Plots/"
    with open(path+config.sim_res_pickle_name, 'rb') as f:
        agents_to_plot_RIGHT, agents_to_plot_LEFT, playerinfo, playerlearning, BOT_avg_expV, MID_avg_expV, TOP_avg_expV, error_bar, expUtility_LEFT, expUtility_RIGHT, allVals = dill.load(f)

    x_RIGHT = np.array([])
    y_RIGHT = np.array([])
    x_LEFT = np.array([])
    y_LEFT = np.array([])


    for i in agents_to_plot_RIGHT:
        i.utility = [x for x in i.utility]# if x != 0]
        x_R_temp = np.full((len(i.utility)), i.get_type()[0])
        x_RIGHT = np.concatenate((x_RIGHT, x_R_temp))
        y_RIGHT = np.concatenate((y_RIGHT, i.utility))


    for i in agents_to_plot_LEFT:
        i.utility = [x for x in i.utility]# if x != 0]
        x_L_temp = np.full((len(i.utility)), i.get_type()[0])
        x_LEFT = np.concatenate((x_LEFT, x_L_temp))
        y_LEFT = np.concatenate((y_LEFT, i.utility))

    # PLOT LEFT AGENTS

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_LEFT,y_LEFT)
    line = slope*x_LEFT+intercept

    fig, ax= plt.subplots()
    ax.plot(x_LEFT,y_LEFT,'o', x_LEFT, line)
    ax.set_xlabel("Agent Type")
    ax.set_ylabel("Payoff")
    title = config.parameters["algo_left"] + " / " + config.parameters["algo_right"] + ". Slope: " + str(slope)
    ax.set_title(title)
    ax.set_xticks(np.arange(min(x_LEFT), max(x_LEFT)+1, 1.0))

    fig.savefig(resPath+config.sim_res_pickle_name[:-4]+" Left.pdf")
    plt.close(fig)

    # PLOT RIGHT AGENTS

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_RIGHT,y_RIGHT)
    line = slope*x_RIGHT+intercept

    fig, ax= plt.subplots()
    ax.plot(x_RIGHT,y_RIGHT,'o', x_RIGHT, line)
    ax.set_xlabel("Agent Type")
    ax.set_ylabel("Payoff")
    title = config.parameters["algo_left"] + " / " + config.parameters["algo_right"] + ". Slope: " + str(slope)
    ax.set_title(title)
    ax.set_xticks(np.arange(min(x_RIGHT), max(x_RIGHT)+1, 1.0))

    fig.savefig(resPath+config.sim_res_pickle_name[:-4]+" Right.pdf")
    plt.close(fig)


def tempPlot(type, threshold_history, Q_learning):
    fig, ax = plt.subplots()
    if Q_learning:
        ax.plot(threshold_history[0])
    else:
        ax.plot(threshold_history)
    ax.set_ylabel("Threshold History")
    ax.set_xlabel("Timestep")
    title = "thresold of "+str(type)
    ax.set_title(title)
    fig.savefig("output/thresold of "+str(type)+".pdf")
    plt.close(fig)