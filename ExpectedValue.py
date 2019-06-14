import numpy as np
import Create_Agents
import random
import config
import Agent
import matplotlib.pyplot as plt
import dill
from statistics import mean, stdev
from collections import defaultdict

def getExpVal(roundCounter, BOT_avg_expV, MID_avg_expV, TOP_avg_expV, list_agents_LEFT, list_agents_RIGHT, num_algo, std, outliers_std):
    # n = 0
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
                elif agent.algo_type == "Qlearn2" or agent.algo_type == "Qlearn" or agent.algo_type == "Wolf":
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
                elif agent.algo_type == "Qlearn2" or agent.algo_type == "Qlearn" or agent.algo_type == "Wolf":
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
            expUtility_LEFT[i]=0
        else:
            # agent can match, calculate expected utility
            expV = mean(match_LEFT[i])
            print "ExpV[", i, ']=', expV
            p = len(match_LEFT[i]) *1.0 / tot_agents_LEFT
            print "p[", i, ']=', p, "(",len(match_LEFT[i]) ,'/',tot_agents_LEFT ,')'
            # n = info_dict_LEFT[i][0][1]
            gamma = info_dict_LEFT[i][0][1]
            if p != 1:
                EU = ( p * (1-p) * expV ) / ( 1 - gamma + (gamma*p) )
            else:
                EU = ( p * expV ) / ( 1 - gamma + (gamma*p) )
            expUtility_LEFT[i]= EU
            print "ExpU[", i, ']=', EU

    for i in range(1,tot_agents_RIGHT+1):
        # calculate all the expected utility for each agent
        if not match_RIGHT[i]:
            # empty, agent cant match with anyone
            expUtility_RIGHT.setdefault(i, 0)
        else:
            # agent can match, calculate expected utility
            expV = mean(match_RIGHT[i])
            p = len(match_RIGHT[i]) *1.0 / tot_agents_RIGHT 
            # n = info_dict_RIGHT[i][0][1]
            gamma = info_dict_RIGHT[i][0][1]
            if p != 1:
                EU = ( p * (1-p) * expV ) / ( 1 - gamma + (gamma*p) )
            else:
                EU = ( p * expV ) / ( 1 - gamma + (gamma*p) )
            expUtility_RIGHT.setdefault(i, EU)


    if num_algo == 2:
        # {agent4 , agent10, agent17}
        expU_outliers = {4: expUtility_LEFT[4], 10: expUtility_LEFT[10], 17: expUtility_LEFT[17]}
        del expUtility_LEFT[4]
        del expUtility_LEFT[10]
        del expUtility_LEFT[17]

    ## calculate the 6 averages
    bottom_avg_left = mean([v for k, v in expUtility_LEFT.items() if k <= 6])
    bottom_avg_right = mean([v for k, v in expUtility_RIGHT.items() if k <= 6])

    mid_avg_left = mean([v for k, v in expUtility_LEFT.items() if (k>6) and (k<=14)])
    mid_avg_right = mean([v for k, v in expUtility_RIGHT.items() if (k>6) and (k<=14)])

    top_avg_left = mean([v for k, v in expUtility_LEFT.items() if k > 14])
    top_avg_right = mean([v for k, v in expUtility_RIGHT.items() if k > 14])

    std[0].append(bottom_avg_left)
    std[1].append(bottom_avg_right)
    std[2].append(mid_avg_left)
    std[3].append(mid_avg_right)
    std[4].append(top_avg_left)
    std[5].append(top_avg_right)

    if num_algo == 2:
        outliers_std[0].append(expU_outliers[17])
        outliers_std[1].append(expU_outliers[10])
        outliers_std[2].append(expU_outliers[4])


    if BOT_avg_expV:
        BOT_avg_expV[0] = (((roundCounter) * BOT_avg_expV[0]) + (bottom_avg_left)) / (roundCounter + 1)
        BOT_avg_expV[1] = (((roundCounter) * BOT_avg_expV[1]) + (bottom_avg_right)) / (roundCounter + 1)

    if MID_avg_expV:
        MID_avg_expV[0] = (((roundCounter) * MID_avg_expV[0]) + (mid_avg_left)) / (roundCounter + 1)
        MID_avg_expV[1] = (((roundCounter) * MID_avg_expV[1]) + (mid_avg_right)) / (roundCounter + 1)

    if TOP_avg_expV:
        TOP_avg_expV[0] = (((roundCounter) * TOP_avg_expV[0]) + (top_avg_left)) / (roundCounter + 1)
        TOP_avg_expV[1] = (((roundCounter) * TOP_avg_expV[1]) + (top_avg_right)) / (roundCounter + 1)
        if num_algo == 2:
            expU_outliers[4] = (((roundCounter) * TOP_avg_expV[2][4]) + (expU_outliers[4])) / (roundCounter + 1)
            expU_outliers[10] = (((roundCounter) * TOP_avg_expV[2][10]) + (expU_outliers[10])) / (roundCounter + 1)
            expU_outliers[17] = (((roundCounter) * TOP_avg_expV[2][17]) + (expU_outliers[17])) / (roundCounter + 1)

            TOP_avg_expV[2] = expU_outliers

    # if avg_expV:
    #     # list is NOT empty, so take the avarege between the 4 just calculated and the 4 passed in
    #     avg_expV[0] = (((roundCounter) * avg_expV[0]) + (top_avg_left)) / (roundCounter + 1)
    #     avg_expV[1] = (((roundCounter) * avg_expV[1]) + (bottom_avg_left)) / (roundCounter + 1)
    #     avg_expV[2] = (((roundCounter) * avg_expV[2]) + (top_avg_right)) / (roundCounter + 1)
    #     avg_expV[3] = (((roundCounter) * avg_expV[3]) + (bottom_avg_right)) / (roundCounter + 1)

    else:
        # list is empty, so just write the values
        BOT_avg_expV.append(bottom_avg_left)
        BOT_avg_expV.append(bottom_avg_right)
        MID_avg_expV.append(mid_avg_left)
        MID_avg_expV.append(mid_avg_right)
        TOP_avg_expV.append(top_avg_left)
        TOP_avg_expV.append(top_avg_right)
        # avg_expV.append(top_avg_left)
        # avg_expV.append(bottom_avg_left)
        # avg_expV.append(top_avg_right)
        # avg_expV.append(bottom_avg_right)
        if num_algo == 2:
            TOP_avg_expV.append(expU_outliers)

    return BOT_avg_expV, MID_avg_expV, TOP_avg_expV, std, outliers_std, expUtility_LEFT, expUtility_RIGHT