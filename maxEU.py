import numpy as np
import Create_Agents
import random
import config
import Agent
import matplotlib.pyplot as plt
import dill
from statistics import mean, stdev
from collections import defaultdict
import glob
import os
from tqdm import tqdm
from itertools import combinations


def ExpVal_AfterLearning(list_agents_LEFT, list_agents_RIGHT):
    learnt_10_EU = []
    expUtility_LEFT = {}
    expUtility_RIGHT = {}
    info_dict_LEFT = {}
    info_dict_RIGHT = {}
    tot_agents_LEFT = len(list_agents_LEFT)
    tot_agents_RIGHT = len(list_agents_RIGHT)
    match_LEFT = defaultdict(lambda:[])
    match_RIGHT = defaultdict(lambda:[])

    for side in ['left', 'right']:
        if side == 'left':
            loop_through = list_agents_LEFT
            other_side = list_agents_RIGHT
        else:
            loop_through = list_agents_RIGHT
            other_side = list_agents_LEFT
        for agent in loop_through:
            # create a dictionary with the relevant information for each agent
            if agent.algo_type == 'Random':
                final_threshold = agent.threshold
            elif agent.algo_type == 'Theoretical':
                final_threshold = agent.threshold.item()
            else:
                final_threshold = agent.threshold_history[-1]
            # n_matched = sum(n+1 for x in agent.meeting_history if (x[2] == 'yes') and (x[3] == 'yes'))
            agent_type = agent.get_type()[0]
            gamma = agent.gamma
            if side == 'left':
                if agent.algo_type == "Model" or agent.algo_type == 'Theoretical' or agent.algo_type == 'Random':
                    info_dict_LEFT.setdefault(agent_type, []).append( (final_threshold, gamma) )
                elif agent.algo_type == "Qlearn2" or agent.algo_type == "Qlearn":
                    if final_threshold[1] == False:
                        poss_match = []
                        for i in other_side:
                            if agent.wouldmatch(i.get_type()):
                                poss_match.append(i.get_type()[0])
                        info_dict_LEFT.setdefault(agent_type, []).append((poss_match, gamma))
                    else:
                        info_dict_LEFT.setdefault(agent_type, []).append( (final_threshold[1], gamma) )
            else:
                if agent.algo_type == "Model" or agent.algo_type == 'Theoretical' or agent.algo_type == 'Random':
                    info_dict_RIGHT.setdefault(agent_type, []).append( (final_threshold, gamma) )
                elif agent.algo_type == "Qlearn2" or agent.algo_type == "Qlearn":
                    if final_threshold[1] == False:
                        poss_match = []
                        for i in other_side:
                            if agent.wouldmatch(i.get_type()):
                                poss_match.append(i.get_type()[0])
                        info_dict_RIGHT.setdefault(agent_type, []).append((poss_match, gamma))
                    else:
                        info_dict_RIGHT.setdefault(agent_type, []).append( (final_threshold[1], gamma) )


    for l_key, l_val in info_dict_LEFT.items():
        for r_key, r_val in info_dict_RIGHT.items():
            thisAgent_matches = []
            if type(r_val[0][0]) == int or type(r_val[0][0]) == float:
                if type(l_val[0][0]) == int or type(l_val[0][0]) == float:
                    # if left agent type >= right agent threshold AND right agent type >= left agent threshold
                    if l_key >= round(r_val[0][0]) and r_key >= round(l_val[0][0]):
                        match_LEFT.setdefault(l_key, thisAgent_matches).append(r_key)
                elif type(l_val[0][0] == list):
                    if l_key >= round(r_val[0][0]) and r_key in l_val[0][0]:
                        match_LEFT.setdefault(l_key, thisAgent_matches).append(r_key)
            elif type(r_val[0][0]) == list:
                if type(l_val[0][0]) == int or type(l_val[0][0]) == float:
                    if l_key in r_val[0][0] and r_key >= round(l_val[0][0]):
                        match_LEFT.setdefault(l_key, thisAgent_matches).append(r_key)
                elif type(l_val[0][0]) == list:
                    if l_key in r_val[0][0] and r_key in l_val[0][0]:
                        match_LEFT.setdefault(l_key, thisAgent_matches).append(r_key)

    for l_key, l_val in info_dict_RIGHT.items():
        for r_key, r_val in info_dict_LEFT.items():
            thisAgent_matches = []
            if type(r_val[0][0]) == int or type(r_val[0][0]) == float:
                if type(l_val[0][0]) == int or type(l_val[0][0]) == float:
                    # if left agent type >= right agent threshold AND right agent type >= left agent threshold
                    if l_key >= round(r_val[0][0]) and r_key >= round(l_val[0][0]):
                        match_RIGHT.setdefault(l_key, thisAgent_matches).append(r_key)
                elif type(l_val[0][0] == list):
                    if l_key >= round(r_val[0][0]) and r_key in l_val[0][0]:
                        match_RIGHT.setdefault(l_key, thisAgent_matches).append(r_key)
            elif type(r_val[0][0]) == list:
                if type(l_val[0][0]) == int or type(l_val[0][0]) == float:
                    if l_key in r_val[0][0] and r_key >= round(l_val[0][0]):
                        match_RIGHT.setdefault(l_key, thisAgent_matches).append(r_key)
                elif type(l_val[0][0]) == list:
                    if l_key in r_val[0][0] and r_key in l_val[0][0]:
                        match_RIGHT.setdefault(l_key, thisAgent_matches).append(r_key)

    for i in range(1,tot_agents_LEFT+1):
        # calculate all the expected utility for each agent
        if not match_LEFT[i]:
            # empty, agent cant match with anyone
            expUtility_LEFT.setdefault(i, 0)
        else:
            # agent can match, calculate expected utility
            expV = mean(match_LEFT[i])
            p = len(match_LEFT[i]) / tot_agents_LEFT
            # n = info_dict_LEFT[i][0][1]
            gamma = info_dict_LEFT[i][0][1]
            if p != 1:
                EU = ( p * (1-p) * expV ) / ( 1 - gamma + (gamma*p) )
            else:
                EU = ( p * expV ) / ( 1 - gamma + (gamma*p) )
            expUtility_LEFT.setdefault(i, EU)

    for i in range(1,tot_agents_RIGHT+1):
        # calculate all the expected utility for each agent
        if not match_RIGHT[i]:
            # empty, agent cant match with anyone
            expUtility_RIGHT.setdefault(i, 0)
        else:
            # agent can match, calculate expected utility
            expV = mean(match_RIGHT[i])
            p = len(match_RIGHT[i]) / tot_agents_RIGHT
            # n = info_dict_RIGHT[i][0][1]
            gamma = info_dict_RIGHT[i][0][1]
            if p != 1:
                EU = ( p * (1-p) * expV ) / ( 1 - gamma + (gamma*p) )
            else:
                EU = ( p * expV ) / ( 1 - gamma + (gamma*p) )
            expUtility_RIGHT.setdefault(i, EU)


    avg_left = mean([v for k, v in expUtility_LEFT.items() if k > howManyAgents])
    avg_right = mean([v for k, v in expUtility_RIGHT.items() if k > howManyAgents])

    learnt_10_EU.append(avg_left)
    learnt_10_EU.append(avg_right)

    return learnt_10_EU


def get_BEST_EU(list_agents_LEFT, list_agents_RIGHT):
    best_10_EU = []
    expUtility_LEFT = {}
    expUtility_RIGHT = {}
    info_dict_LEFT = {}
    info_dict_RIGHT = {}
    tot_agents_LEFT = len(list_agents_LEFT)
    tot_agents_RIGHT = len(list_agents_RIGHT)
    match_LEFT = defaultdict(lambda:[])
    match_RIGHT = defaultdict(lambda:[])

    for side in ['left', 'right']:
        if side == 'left':
            loop_through = list_agents_LEFT
            other_side = list_agents_RIGHT
        else:
            loop_through = list_agents_RIGHT
            other_side = list_agents_LEFT
        for agent in loop_through:
            # create a dictionary with the relevant information for each agent
            if agent.algo_type == 'Random':
                final_threshold = agent.threshold
            elif agent.algo_type == 'Theoretical':
                final_threshold = agent.threshold.item()
            else:
                final_threshold = agent.threshold_history[-1]
            # n_matched = sum(n+1 for x in agent.meeting_history if (x[2] == 'yes') and (x[3] == 'yes'))
            agent_type = agent.get_type()[0]
            gamma = agent.gamma
            if side == 'left':
                if agent.algo_type == "Model" or agent.algo_type == 'Theoretical' or agent.algo_type == 'Random':
                    info_dict_LEFT.setdefault(agent_type, []).append( (final_threshold, gamma) )
                elif agent.algo_type == "Qlearn2" or agent.algo_type == "Qlearn":
                    if final_threshold[1] == False:
                        poss_match = []
                        for i in other_side:
                            if agent.wouldmatch(i.get_type()):
                                poss_match.append(i.get_type()[0])
                        info_dict_LEFT.setdefault(agent_type, []).append((poss_match, gamma))
                    else:
                        info_dict_LEFT.setdefault(agent_type, []).append( (final_threshold[1], gamma) )
            else:
                if agent.algo_type == "Model" or agent.algo_type == 'Theoretical' or agent.algo_type == 'Random':
                    info_dict_RIGHT.setdefault(agent_type, []).append( (final_threshold, gamma) )
                elif agent.algo_type == "Qlearn2" or agent.algo_type == "Qlearn":
                    if final_threshold[1] == False:
                        poss_match = []
                        for i in other_side:
                            if agent.wouldmatch(i.get_type()):
                                poss_match.append(i.get_type()[0])
                        info_dict_RIGHT.setdefault(agent_type, []).append((poss_match, gamma))
                    else:
                        info_dict_RIGHT.setdefault(agent_type, []).append( (final_threshold[1], gamma) )


    for l_key, l_val in info_dict_LEFT.items():
        for r_key, r_val in info_dict_RIGHT.items():
            thisAgent_matches = []
            if type(r_val[0][0]) == int or type(r_val[0][0]) == float:
                if type(l_val[0][0]) == int or type(l_val[0][0]) == float:
                    # if left agent type >= right agent threshold AND right agent type >= left agent threshold
                    if l_key >= round(r_val[0][0]):
                        match_LEFT.setdefault(l_key, thisAgent_matches).append(r_key)
                elif type(l_val[0][0] == list):
                    if l_key >= round(r_val[0][0]):
                        match_LEFT.setdefault(l_key, thisAgent_matches).append(r_key)
            elif type(r_val[0][0]) == list:
                if type(l_val[0][0]) == int or type(l_val[0][0]) == float:
                    if l_key in r_val[0][0]:
                        match_LEFT.setdefault(l_key, thisAgent_matches).append(r_key)
                elif type(l_val[0][0]) == list:
                    if l_key in r_val[0][0]:
                        match_LEFT.setdefault(l_key, thisAgent_matches).append(r_key)

    for l_key, l_val in info_dict_RIGHT.items():
        for r_key, r_val in info_dict_LEFT.items():
            thisAgent_matches = []
            if type(r_val[0][0]) == int or type(r_val[0][0]) == float:
                if type(l_val[0][0]) == int or type(l_val[0][0]) == float:
                    # if left agent type >= right agent threshold AND right agent type >= left agent threshold
                    if l_key >= round(r_val[0][0]):
                        match_RIGHT.setdefault(l_key, thisAgent_matches).append(r_key)
                elif type(l_val[0][0] == list):
                    if l_key >= round(r_val[0][0]):
                        match_RIGHT.setdefault(l_key, thisAgent_matches).append(r_key)
            elif type(r_val[0][0]) == list:
                if type(l_val[0][0]) == int or type(l_val[0][0]) == float:
                    if l_key in r_val[0][0]:
                        match_RIGHT.setdefault(l_key, thisAgent_matches).append(r_key)
                elif type(l_val[0][0]) == list:
                    if l_key in r_val[0][0]:
                        match_RIGHT.setdefault(l_key, thisAgent_matches).append(r_key)

    for i in range(1,tot_agents_LEFT+1):
        # calculate all the expected utility for each agent
        if not match_LEFT[i]:
            # empty, agent cant match with anyone
            expUtility_LEFT.setdefault(i, 0)
        else:
            agentsThatCanMatch = sorted(match_LEFT[i], reverse=True)
            permutationAgents = []
            optimalEU = -1
            for j, agentType in enumerate(agentsThatCanMatch):
                permutationAgents.append(agentType)
                expV = mean(permutationAgents)
                p = len(permutationAgents) / tot_agents_LEFT
                gamma = info_dict_LEFT[i][0][1]
                if p != 1:
                    EU = ( p * (1-p) * expV ) / ( 1 - gamma + (gamma*p) )
                else:
                    EU = ( p * expV ) / ( 1 - gamma + (gamma*p) )
                optimalEU = max(optimalEU, EU)
            expUtility_LEFT.setdefault(i, optimalEU)

    for i in range(1,tot_agents_RIGHT+1):
        # calculate all the expected utility for each agent
        if not match_RIGHT[i]:
            # empty, agent cant match with anyone
            expUtility_RIGHT.setdefault(i, 0)
        else:
            agentsThatCanMatch = sorted(match_RIGHT[i], reverse=True)
            permutationAgents = []
            optimalEU = -1
            for j, agentType in enumerate(agentsThatCanMatch):
                permutationAgents.append(agentType)
                expV = mean(permutationAgents)
                p = len(permutationAgents) / tot_agents_RIGHT
                gamma = info_dict_RIGHT[i][0][1]
                if p != 1:
                    EU = ( p * (1-p) * expV ) / ( 1 - gamma + (gamma*p) )
                else:
                    EU = ( p * expV ) / ( 1 - gamma + (gamma*p) )
                optimalEU = max(optimalEU, EU)
            expUtility_RIGHT.setdefault(i, optimalEU)


    avg_left = mean([v for k, v in expUtility_LEFT.items() if k > howManyAgents])
    avg_right = mean([v for k, v in expUtility_RIGHT.items() if k > howManyAgents])

    best_10_EU.append(avg_left)
    best_10_EU.append(avg_right)

    return best_10_EU



whichfolder = "Top_10_Agents"
howManyAgents = 9
# whichfolder = "All_Agents"
# howManyAgents = 0
gammaVal = 75
path = "./../all_results/maxEU/"+whichfolder+"/Gamma"+str(gammaVal)+"/"
cd=os.getcwd()
os.chdir(path)
open('./_ALLmaxEU.txt', 'w').close()
open('./_maxEU.txt', 'w').close()

qlearners = []
modellers = []
q_l =[]
q_l_e = []
knn = []
epsilon = []

for file in tqdm(glob.glob("*.pkl")):
    name_left = file[:file.rfind(' vs ')]
    name_right = file[file.rfind(' vs ')+4:-4]
    try:
        with open(str(file), 'rb') as f:
            _, _, _, _, _, _, _, _, _, _, allVals = dill.load(f)
            pl = 4
            pr = 5
    except:
        with open(str(file), 'rb') as f:
            _, _, _, _, _, _, _, _, allVals = dill.load(f)
            pl = 2
            pr = 3

    percentageLeft = []
    percentageRight = []
    for i in range(len(allVals)):
        players_left = allVals[i][pl]
        players_right = allVals[i][pr]
        learnt_10_EU = ExpVal_AfterLearning(players_left, players_right)
        bestEU = get_BEST_EU(players_left, players_right)
        percentageLeft.append( (learnt_10_EU[0]*100)/bestEU[0]  )
        percentageRight.append( (learnt_10_EU[1]*100)/bestEU[1]  )

    if name_left[:9] == 'QLearning':
        qlearners.append(mean(percentageLeft))
    else:
        modellers.append(mean(percentageLeft))
    if name_right[:9] == 'QLearning':
        qlearners.append(mean(percentageRight))
    else:
        modellers.append(mean(percentageRight))

    if name_left == 'QLearning2':
        q_l.append(mean(percentageLeft))
    elif name_left == 'QLearning2 EXPLORATION':
        q_l_e.append(mean(percentageLeft))
    elif name_left == 'KNN':
        knn.append(mean(percentageLeft))
    elif name_left == 'EPSILON_THRESHOLD':
        epsilon.append(mean(percentageLeft))

    if name_right == 'QLearning2':
        q_l.append(mean(percentageRight))
    elif name_right == 'QLearning2 EXPLORATION':
        q_l_e.append(mean(percentageRight))
    elif name_right == 'KNN':
        knn.append(mean(percentageRight))
    elif name_right == 'EPSILON_THRESHOLD':
        epsilon.append(mean(percentageRight))


    with open("./_ALLmaxEU.txt", "a") as myfile:
        myfile.write("\n\n" + name_left + " vs " + name_right + ":\t"+ str(round(mean(percentageLeft), 2)))
    with open("./_ALLmaxEU.txt", "a") as myfile:
        myfile.write("\n\n" + name_right + " vs " + name_left + ":\t"+ str(round(mean(percentageRight), 2)))


with open("./_maxEU.txt", "a") as myfile:
        myfile.write(whichfolder+"\nGAMMA: "+str(gammaVal)+"\n\nModellers:\t" + str(round(mean(modellers), 2)) + "\nQlearners:\t" +str(round(mean(qlearners), 2)) + "\n\n_____________________\n\nQLearning2:\t\t" +  str(round(mean(q_l), 2)) + "\nQLearning2 EXPLORATION:\t" +  str(round(mean(q_l_e), 2))+ "\nKNN:\t\t\t" +  str(round(mean(knn), 2))+ "\nEPSILON_THRESHOLD:\t" +  str(round(mean(epsilon), 2)) )