import numpy as np
import Create_Agents
import random
import config
import Agent
import matplotlib.pyplot as plt
import dill
import Plot
import time
from tqdm import tqdm
from statistics import mean, stdev
import ExpectedValue
import Plot_Histograms


BOT_avg_expV = []
MID_avg_expV = []
TOP_avg_expV = []
std = [[], [], [], [], [], []]
outliers_std = [[], [], []]
error_bar = []
expUtility_LEFT = {}
expUtility_RIGHT = {}
allVals = []

threshold_for_analysis_storage = {}
rank_analysis_storage = {}
opponent_analysis_storage = {}
players_left_storage = {}
players_right_storage = {}
repeated_storage = {}
opp_dist_for_analysis_storage = {}
opp_answer_dist_for_analysis_storage = {}
opp_yes_prob_for_analysis_storage = {}
final_utility = []


# for each round
print(config.algo_left[0]," vs ", config.algo_right[0])
for simulation in tqdm(range(config.simulation_rounds)):
# for simulation in range(config.simulation_rounds):
    players_left, players_right, Create_Agents.Left_Ranked_Attractiveness, Create_Agents.Right_Ranked_Attractiveness = Create_Agents.gen_instancelist()
    threshold_for_analysis = {}
    rank_analysis = {}
    opp_dist_for_analysis = {}
    opp_answer_dist_for_analysis = {}
    opp_yes_prob_for_analysis = {}
    opponent_analysis = {}
    strategy_scores = {}
    set_of_matchings = []
    # ------------
    agents_to_plot_LEFT = []
    agents_to_plot_RIGHT = []
    # ------------
    # for each step
    for repeated in tqdm(range(config.simulation_steps)):

        # just print the iteration counter
        if repeated%10==0 and not config.verbose_silentSimulations:
            if repeated==0: print("Repeated = {0}".format(repeated+1))
            else: print("Repeated = {0}".format(repeated))

        set_of_meetings = []

        # randomly shuffle left set
        random.shuffle(players_left)
            # find a meeting for each 'Left' players
        # create a list of tuples matching first left player with first right player, etc...
        meetings = list(zip(players_left, players_right))

        for meeting in meetings:
            main_player, potential_match = meeting[0], meeting[1]
            # get decision for one player
            main_player_decision, main_player_utility = main_player.get_decision(potential_match)
            # get decision for the other player
            potential_match_decision, potential_match_utility = potential_match.get_decision(main_player)
            # if they both wanna match
            if main_player_decision == "yes" and potential_match_decision == "yes":
                set_of_matchings.append((main_player.get_type(), main_player_utility, potential_match.get_type(), potential_match_utility))
                main_player.notify_matched(potential_match)
                potential_match.notify_matched(main_player)
                strategy_scores.setdefault(main_player.strategy, []).append(main_player_utility)
                strategy_scores.setdefault(potential_match.strategy, []).append(potential_match_utility)
            if repeated == config.simulation_steps-1:
                final_utility.append(main_player_utility + potential_match_utility)

    if config.verbose_analysis:
                # print("  |number_of_matched_players| = {0}".format(len(set_of_matchings)))
                # print("  |matched_players| = {0}".format(set_of_matchings))
        print("final_utility: {0}".format(sum(final_utility)))


    allwouldmatches = []


    for agentleft in players_left:
        for agentright in players_right:
        #determine whether agentleft would match with agentright AND agentright would match with agentleft
            if (agentleft.wouldmatch(agentright.get_type()) and agentright.wouldmatch(agentleft.get_type())):
                allwouldmatches.append((agentleft.get_type()[0], agentright.get_type()[0]))

    print allwouldmatches
    with open('output/Pickles/'+config.sim_res_pickle_name, 'w') as f:
        f.write('\t'.join([str(p1)+','+str(p2) for p1,p2 in allwouldmatches])+'\n')

#    playerinfo = []
 #   playerlearning = []
    # -------------------------------
    # -------------------------------
#    for strategy in config.strategies:
#        siminfo = strategy+ "_"+str(config.simulation_steps)+"rounds_rep"+str(simulation)

#        for side in ['left', 'right']:
#            if side == 'left':
#                players = players_left
#            else:
#                players = players_right
#
#            # skip everything else if no agents on left/right have this strategy
#            if len([p for p in players if p.strategy==strategy]) ==0: # if no agents on left/right have this strategy
#                continue
#
#            # file_matches = open("output/results_matches_" + siminfo + '_' + side + ".tsv", "w")
 #           # file_learning = open("output/results__learning_" + siminfo + '_' + side + ".tsv", "w")

            # loop through each agent in the left/right player set which has that particular strategy
#            for agent in [p for p in players if p.strategy==strategy]:
#                playerinfo = [agent.get_type()[0]] + agent.meeting_history
#                playerlearning = [agent.get_type()[0]] + agent.threshold_history

                # if side == 'left':
                #     agents_to_plot_LEFT.append(agent)
                # else:
                #     agents_to_plot_RIGHT.append(agent)

                # if side == 'left' and agent.get_type()[0] == 2:
                #     Plot.tempPlot(2, agent.threshold_history, False)
                # elif side == 'left' and agent.get_type()[0] == 6:
                #     Plot.tempPlot(6, agent.threshold_history, False)
                # elif side == 'left' and agent.get_type()[0] == 10:
                #     Plot.tempPlot(10, agent.threshold_history, False)
                # elif side == 'left' and agent.get_type()[0] == 17:
                #     Plot.tempPlot(17, agent.threshold_history, False)

                # file_matches.write("\t".join([str(i) for i in playerinfo]) + "\n")
                # file_learning.write("\t".join([str(i) for i in playerlearning]) + "\n")
                # plot threshold agent 2, 10, 17
#    if len(config.algo_left) == 1:
#        BOT_avg_expV, MID_avg_expV, TOP_avg_expV, std, outliers_std, expUtility_LEFT, expUtility_RIGHT = ExpectedValue.getExpVal(simulation, BOT_avg_expV, MID_avg_expV, TOP_avg_expV, players_left, players_right, 1, std, outliers_std)
#        allVals.append([simulation, BOT_avg_expV, MID_avg_expV, TOP_avg_expV, players_left, players_right, 1, std, outliers_std])
#    elif len(config.algo_left) == 2:
#        BOT_avg_expV, MID_avg_expV, TOP_avg_expV, std, outliers_std, expUtility_LEFT, expUtility_RIGHT = ExpectedValue.getExpVal(simulation, BOT_avg_expV, MID_avg_expV, TOP_avg_expV, players_left, players_right, 2, std, outliers_std)
#        allVals.append([simulation, BOT_avg_expV, MID_avg_expV, TOP_avg_expV, players_left, players_right, 2, std, outliers_std])

#error_bar.append(stdev(std[0]) / (len(std[0]) ** 0.5))
#error_bar.append(stdev(std[1]) / (len(std[1]) ** 0.5))
#error_bar.append(stdev(std[2]) / (len(std[2]) ** 0.5))
#error_bar.append(stdev(std[3]) / (len(std[3]) ** 0.5))
#error_bar.append(stdev(std[4]) / (len(std[3]) ** 0.5))
#error_bar.append(stdev(std[5]) / (len(std[3]) ** 0.5))
#if len(config.algo_left) == 2:
    ##     [bottom_avg_left, bottom_avg_right, mid_avg_left, mid_avg_right, top_avg_left, top_avg_right, 4, 10, 17]
#    error_bar.append(stdev(outliers_std[0]) / (len(outliers_std[0]) ** 0.5))
#    error_bar.append(stdev(outliers_std[1]) / (len(outliers_std[1]) ** 0.5))
#    error_bar.append(stdev(outliers_std[2]) / (len(outliers_std[2]) ** 0.5))

#with open('output/Pickles/'+config.sim_res_pickle_name, 'wb') as f:
#    dill.dump([players_right, players_left, playerinfo, playerlearning, BOT_avg_expV, MID_avg_expV, TOP_avg_expV, error_bar, expUtility_LEFT, expUtility_RIGHT, allVals], f)


#error_bar = [round(x, 2) for x in error_bar]
#BOT_avg_expV = [round(x, 2) for x in BOT_avg_expV]
#MID_avg_expV = [round(x, 2) for x in MID_avg_expV]
#TOP_avg_expV[0] = round(TOP_avg_expV[0], 2)
#TOP_avg_expV[1] = round(TOP_avg_expV[1], 2)
#if len(config.algo_left) == 2:
#    TOP_avg_expV[2][4] = round(TOP_avg_expV[2][4], 2)
#    TOP_avg_expV[2][10] = round(TOP_avg_expV[2][10], 2)
#    TOP_avg_expV[2][17] = round(TOP_avg_expV[2][17], 2)

#with open("output/results.txt", "a") as myfile:
#    myfile.write("\n\n"+config.output_notes+": \n"+ "BOTTOM expV: \t" + str(BOT_avg_expV) + "\nMID expV: \t" + str(MID_avg_expV) + "\nTOP expV: \t" + str(TOP_avg_expV) +  " \n "+str(error_bar)+ "\n_______________________________________\n")

#Plot.plotRes()
#Plot_Histograms.makeHist()