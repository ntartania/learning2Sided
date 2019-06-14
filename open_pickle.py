import config
import matplotlib.pyplot as plt
import numpy as np

def open_pickle(run, learnfile, matchfile):
    nine_element = 1
    one_side_only = 0
    opp_yes_prob_for_analysis = None
    # file = 'output_100'
    # file = config.output_name

    threshlearning = {}
    with open(learnfile) as f:
        for line in f:
            alldata = line.split("\t")
            threshlearning[alldata[0]] = alldata[1:]

    with open(matchfile) as f:
        for line in f:
            alldata = line.split("\t")
            threshlearning[alldata[0]] = alldata[1:]


    # with open(file+'_backup') as f:
    #     threshold_for_analysis, rank_analysis, opponent_analysis, players_left, players_right, repeat = pickle.load(f)
    #
    # with open(file+'theory'+'_backup') as f:
    #     threshold_l, threshold_r = pickle.load(f)

    # print 'threshold_l', threshold_l
    # print threshold_for_analysis
    #
    # actual_threshold = [(i+1, round(threshold_for_analysis[t][-1],2)) for i, t in enumerate(threshold_for_analysis)]
    # theory_threshold_left = [(i+1, round(threshold_l[t],2)) for i, t in enumerate(threshold_l)]
    # theory_threshold_right = [(config.number_of_players_left+i+1, round(threshold_r[t],2)) for i, t in enumerate(threshold_r)]
    # theory_threshold = theory_threshold_left + theory_threshold_right
    #
    # print actual_threshold
    # print theory_threshold
    # print 'opp_dist_for_analysis',opp_dist_for_analysis
    # print 'opp_answer_dist_for_analysis',opp_answer_dist_for_analysis
    # if opp_yes_prob_for_analysis:
    #     print 'opp_yes_prob_for_analysis', opp_yes_prob_for_analysis
    # print 'repeat',repeat


    # print threshold_for_analysis
    # print len(threshold_for_analysis[1])
    # print threshold_l
    # print len(threshold_l)
    # print repeat
    # exit()


    # actual_threshold_l = {rank_analysis[ID][0]: threshold_for_analysis[ID] for ID in range(1, config.number_of_players_left+1)}
    # actual_threshold_r = {rank_analysis[ID][0]: threshold_for_analysis[ID] for ID in range(config.number_of_players_left+1, config.number_of_players_left + config.number_of_players_right+1)}
    # threshold_error_l = {i: [abs(actual_threshold_l[i][index] - threshold_l[i]) for index in range(repeat)] for i in threshold_l}
    # threshold_error_r = {i: [abs(actual_threshold_r[i][index] - threshold_r[i]) for index in range(repeat)] for i in threshold_r}


    # MAE_l = np.mean(threshold_error_l)
    # MAE_r = np.mean(threshold_error_r)
    # print 'actual_threshold_l', actual_threshold_l
    # # print 'theorylthreshold_l', threshold_l
    # print 'actual_threshold_r', actual_threshold_r
    # # print 'theorylthreshold_r', threshold_r
    # print 'threshold_error_l', threshold_error_l[1]
    # print 'threshold_error_r', threshold_error_r[1]
    # # print 'MAE_l', MAE_l
    # # print 'MAE_r', MAE_r
    # print 'opp_dist_for_analysis', opp_dist_for_analysis
    # print 'opp_answer_dist_for_analysis', opp_answer_dist_for_analysis
    # print repeat

    # exit()

    # print [(i+1, actual_threshold[i][1], theory_threshold[i][1]) for i in range(len(actual_threshold))]

    # print threshold_for_analysis
    # print rank_analysis
    # # exit()
    #
    # print len(threshold_for_analysis[1])
    # print rank_analysis
    # print opponent_analysis
    # exit()

    # plotting
    if config.plotting == 1:
    # rank matching plot
        plt.figure(1)
        plt.cla() # Clear axis
        plt.clf() # Clear figure
        plt.title("Rank Matching from LEFT to RIGHT")
        X, Y = [], []
        for player in players_left:
            if opponent_analysis[player.ID]:
                X.append(rank_analysis[player.ID][0])
                Y.append(opponent_analysis[player.ID][0])

        plt.plot(X, Y, 'bo')
        plt.plot([0, config.number_of_players], [0, config.number_of_players], 'g--')
        plt.axis([1.0*0.85, config.number_of_players * 1.01, 1.0*0.85, config.number_of_players * 1.01])
        plt.xlabel("Rank of Agents on the LEFT")
        plt.ylabel("Matched Rank of Agents on the RIGHT")
        plt.savefig(output_name + '/' + output_name + '_1.pdf')
        # plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
        # plt.axis([0.5, 10.5, 0.5, 10.5])
        # plt.show()

        plt.figure(2)
        plt.cla()
        plt.clf()
        if one_side_only == 0:
            plt.title("Rank Matching from RIGHT to LEFT")
            X, Y = [], []
            for player in players_right:
                if opponent_analysis[player.ID]:
                    X.append(rank_analysis[player.ID][0])
                    Y.append(opponent_analysis[player.ID][0])

            plt.plot(X, Y, 'ro')
            plt.plot([0, config.number_of_players], [0, config.number_of_players], 'g--')
            plt.axis([1.0*0.85, config.number_of_players * 1.01, 1.0*0.85, config.number_of_players * 1.01])
            plt.xlabel("Rank of Agents on the RIGHT")
            plt.ylabel("Matched Rank of Agents on the LEFT")
        plt.savefig(output_name + '/' + output_name + '_2.pdf')
        # plt.axis([0.5, 10.5, 0.5, 10.5])
        # plt.show()

        # threshold plotting
        plt.figure(3)
        plt.cla()
        plt.clf()
        plt.title("Threshold Development over Time")
        X, Y = [x+1 for x in range(len(threshold_for_analysis[1]))], []
        cmap = plt.get_cmap('jet_r')
        # print threshold_for_analysis
        # print 'X', X
        # exit()
        for i, player in enumerate(players_left[::-1]):
            color = cmap(float(i) / len(players_left))
            # if isinstance(rank_analysis[player.ID][1], int) and isinstance(rank_analysis[player.ID][3], int):
            if 1:
                if rank_analysis[player.ID][0] % config.rank_analysis_mod == 0:
                    Y = [round(t, config.random_decimal_point) for t in threshold_for_analysis[player.ID]]
                    # print 'thresh', threshold_for_analysis[player.ID]
                    # print 'rank', rank_analysis[player.ID]
                    # print 'Y', Y
                    # if threshold_error_l >= MAE_l:
                    if config.number_of_players <= 5:
                        plt.plot(X, Y, '-', label='$rank = {rank}$'.format(rank=rank_analysis[player.ID][0]), c=color)
                    else:
                        if rank_analysis[player.ID][0] == 1 or rank_analysis[player.ID][0] == config.number_of_players or rank_analysis[player.ID][0]%3 == 0:
                            plt.plot(X, Y, '-', label='$rank = {rank}$'.format(rank=rank_analysis[player.ID][0]), c=color)
                        else:
                            plt.plot(X, Y, '-', c=color)
                    # plt.plot(X, Y, '-', label='_nolegend_'.format(rank=rank_analysis[player.ID][0]), c=color)
                    plt.plot([0, len(threshold_for_analysis[1])], [threshold_l[rank_analysis[player.ID][0]], threshold_l[rank_analysis[player.ID][0]]], ls = '--', c=color)

        plt.legend(loc='lower left', framealpha=0.6)
        # plt.plot([0, len(threshold_for_analysis[1])], [1, 1], 'b--')        # need normalisation
        # plt.plot([0, len(threshold_for_analysis[1])], [0, 0], 'r--')        # need normalisation
        # plt.plot([0, len(threshold_for_analysis[1])], [-1, -1], 'b--')      # need normalisation
        # plt.show()
        # exit()
        # plt.axis([-0.5, repeat+0.5, -0.5, 30.5])
        plt.axis([0, repeat, 0.0, config.number_of_players * 1.01])
        plt.xlabel("Number of Round")
        plt.ylabel("Discounted Utility Value\n(Dash Lines: Ideal Threshold for each Agents)")
        plt.savefig(output_name + '/' + output_name + '_3.pdf')

        plt.figure(4)
        plt.cla()
        plt.clf()
        if one_side_only == 0:
            plt.title("Threshold Development over Time")
            X, Y = [x + 1 for x in range(len(threshold_for_analysis[1]))], []
            cmap = plt.get_cmap('jet_r')  # print threshold_for_analysis
            # print 'X', X
            # exit()
            for i, player in enumerate(players_right[::-1]):
                color = cmap(float(i) / len(players_right))
                # if isinstance(rank_analysis[player.ID][1], int) and isinstance(rank_analysis[player.ID][3], int):
                if 1:
                    if rank_analysis[player.ID][0] % config.rank_analysis_mod == 0:
                        Y = [round(t, config.random_decimal_point) for t in
                             threshold_for_analysis[player.ID]]
                        # print 'thresh', threshold_for_analysis[player.ID]
                        # print 'rank', rank_analysis[player.ID]
                        # print 'Y', Y
                        if config.number_of_players <= 5:
                            plt.plot(X, Y, '-', label='$rank = {rank}$'.format(rank=rank_analysis[player.ID][0]), c=color)
                        else:
                            if rank_analysis[player.ID][0] == 1 or rank_analysis[player.ID][0] == config.number_of_players or rank_analysis[player.ID][0]%3 == 0:
                                plt.plot(X, Y, '-', label='$rank = {rank}$'.format(rank=rank_analysis[player.ID][0]), c=color)
                            else:
                                plt.plot(X, Y, '-', c=color)
                        plt.plot([0, len(threshold_for_analysis[1])],[threshold_r[rank_analysis[player.ID][0]], threshold_r[rank_analysis[player.ID][0]]], ls='--',c=color)
            plt.legend(loc='lower left', framealpha=0.6)
            # plt.plot([0, len(threshold_for_analysis[1])], [1, 1], 'b--')        # need normalisation
            # plt.plot([0, len(threshold_for_analysis[1])], [0, 0], 'r--')        # need normalisation
            # plt.plot([0, len(threshold_for_analysis[1])], [-1, -1], 'b--')      # need normalisation
            # plt.show()
            # exit()
            # plt.axis([-0.5, repeat+0.5, -0.5, 30.5])
            plt.axis([0, repeat, 0.0, config.number_of_players * 1.01])
            plt.xlabel("Number of Round")
            plt.ylabel("Discounted Utility Value\n(Dash Lines: Ideal Threshold for each Agents)")
        plt.savefig(output_name + '/' + output_name + '_4.pdf')
        # snake ladder plotting
        plt.figure(5)
        plt.cla()
        plt.clf()
        plt.title("Matching Proposed from LEFT to RIGHT")
        X, Y = [], []

        for player in players_left:
            if opponent_analysis[player.ID]:
                plt.plot([0, 1], [rank_analysis[player.ID][0], opponent_analysis[player.ID][0]], 'b-')
                # plt.plot([0, 1], [rank_analysis[player.ID][0], opponent_analysis[player.ID][0]], 'b--', linewidth=2.0)

        for painter in range(config.number_of_players):
            plt.plot([0, 0], [painter+1, painter+1], 'bo')
            plt.plot([1, 1], [painter+1, painter+1], 'ro')

        plt.axis([-0.2, 1.2, 0.5, config.number_of_players + 0.5])
        plt.xlabel("Set of Agents")
        plt.ylabel("Player Rank\n(Rank 1: Highest Valued)")

        plt.xticks([0, 1], ["LEFT", "RIGHT"])
        plt.savefig(output_name + '/' + output_name + '_5.pdf')
        # plt.show()

        plt.figure(6)
        plt.cla()
        plt.clf()
        if one_side_only == 0:
            plt.title("Matching Proposed from RIGHT to LEFT")
            X, Y = [], []
            for player in players_right:
                if opponent_analysis[player.ID]:
                    plt.plot([0, 1], [opponent_analysis[player.ID][0], rank_analysis[player.ID][0]], 'r-')

            for painter in range(config.number_of_players):
                plt.plot([0, 0], [painter+1, painter+1], 'bo')
                plt.plot([1, 1], [painter+1, painter+1], 'ro')

            plt.axis([-0.2, 1.2, 0.5, config.number_of_players + 0.5])
            plt.xlabel("Set of Agents")
            plt.ylabel("Player Rank\n(Rank 1: Highest Valued)")

            plt.xticks([0,1], ["LEFT","RIGHT"])
        plt.savefig(output_name + '/' + output_name + '_6.pdf')
        # plt.show()

        plt.figure(7)
        plt.cla()
        plt.clf()
        plt.title("Mean Absolute Error (MAE)")
        X, Y, Z = [x for x in range(1, repeat+1)], [], []

        # print range(1, config.number_of_players_left+1)
        # print threshold_error_l

        if one_side_only == 0:
            Y = [np.average([threshold_error_l[plyr_rnk][rnd-1] for plyr_rnk in range(1, config.number_of_players_left + 1)]) for rnd in X]
            Z = [np.average([threshold_error_r[plyr_rnk][rnd-1] for plyr_rnk in range(1, config.number_of_players_right + 1)]) for rnd in X]  # print Y
            plt.plot(X, Y, 'b', label="Agents on the LEFT,\nFinal MAE = {0}".format(round(Y[-1],2)))
            plt.plot(X, Z, 'r', label="Agents on the RIGHT,\nFinal MAE = {0}".format(round(Z[-1],2)))
            plt.legend(loc="upper right", framealpha=0.6)
            plt.xlabel("Number of Round")
            plt.ylabel("Mean Absolute Error")
            if output_name in set_of_interest:
                mae[output_name] = (Y,Z)

        else:
            Y = [np.average([threshold_error_l[plyr_rnk][rnd-1] for plyr_rnk in range(1, config.number_of_players_left + 1)]) for rnd in X]
            Z = [np.average([threshold_error_r[plyr_rnk][rnd-1] for plyr_rnk in range(1, config.number_of_players_right + 1)]) for rnd in X]  # print Y
            YZ = [float(Y[runner] + Z[runner])/2 for runner in range(len(Y))]
            if output_name in set_of_interest:
                mae[output_name] = YZ
                print(mae)
            plt.plot(X, YZ, 'b', label="Final MAE = {0}".format(round(YZ[-1], 2)))
            plt.legend(loc="upper right", framealpha=0.6)
            plt.xlabel("Number of Round")
            plt.ylabel("Mean Absolute Error")


        plt.savefig(output_name + '/' + output_name + '_7.pdf')

        plt.figure(8)
        plt.cla()
        plt.clf()
        plt.title("Convergence Rate")
        X, Y = [x for x in range(1, repeat+1)], []

        threshold_to_count_as_converge = 0.5

        plt.plot([0, repeat],[config.number_of_players, config.number_of_players], ls='--', c='green', label="Baseline for Perfect Convergence")

        if one_side_only == 0:
            Y = [np.sum([1.0 for plyr_rnk in range(1, config.number_of_players_left + 1) if threshold_error_l[plyr_rnk][rnd - 1] <= threshold_to_count_as_converge]) for rnd in X]
            Z = [np.sum([1.0 for plyr_rnk in range(1, config.number_of_players_right + 1) if threshold_error_r[plyr_rnk][rnd - 1] <= threshold_to_count_as_converge]) for rnd in X]

            plt.plot(X, Y, 'b', label="Converge Count on the LEFT,\nConverge Rt. = {0}".format(round(Y[-1]/config.number_of_players,2)))
            plt.plot(X, Z, 'r', label="Converge Count on the RIGHT,\nConverge Rt. = {0}".format(round(Z[-1]/config.number_of_players,2)))
            if output_name in set_of_interest:
                converge_rate[output_name] = (Y, Z)
        else:
            Y = [np.sum([1.0 for plyr_rnk in range(1, config.number_of_players_left + 1) if threshold_error_l[plyr_rnk][rnd - 1] <= threshold_to_count_as_converge]) for rnd in X]
            Z = [np.sum([1.0 for plyr_rnk in range(1, config.number_of_players_right + 1) if threshold_error_r[plyr_rnk][rnd - 1] <= threshold_to_count_as_converge]) for rnd in X]
            YZ = [float(Y[runner] + Z[runner])/2 for runner in range(len(Y))]
            if output_name in set_of_interest:
                converge_rate[output_name] = YZ
                print(converge_rate)
            plt.plot(X, YZ, 'b', label="Converge Rt. = {0}".format(round(YZ[-1] / config.number_of_players, 2)))

        if output_name not in ["g95", "a02", "1nn_exp4", "1nn_exp3", "1nn_exp2", "1nn_exp1", "exp4", "exp3", "exp2", "exp1" ]:
            plt.legend(loc="lower right", framealpha=0.6)
        else:
            plt.legend(loc="upper left", framealpha=0.6)
        plt.xlabel("Number of Round")
        plt.ylabel("Number of Converged Agents (Max = {0})".format(config.number_of_players))
        plt.axis([0, repeat, 0.0, config.number_of_players * 1.01])

        plt.savefig(output_name + '/' + output_name + '_8.pdf')
        # plt.show()

    return mae, converge_rate

    # wanna_see_rank = 5
    # picked = [i for i in rank_analysis if rank_analysis[i][0] == wanna_see_rank]
    # for item in picked:
    #     print [round(num,2) for num in threshold_for_analysis[item]]
    #     print rank_analysis[item]

if __name__ == "__main__":
    mae, converge_rate = {}, {}
    one_side_only = 0
    # set_of_interest = ["Std","exp1", "exp2", "exp3", "exp4"]
    # full_name = {"Std":"No exploration","exp1":"Non-optimal exploration", "exp2":"Fixed Yes-No exploration", "exp3":"Yes-to-unknown exploration", "exp4":"Yes-to-percentage-unknown exploration"}
    # set_of_interest = ["1nn", "1nn_exp1", "1nn_exp2", "1nn_exp3", "1nn_exp4"]
    # full_name = {"1nn": "No exploration", "1nn_exp1": "Non-optimal exploration", "1nn_exp2": "Fixed Yes-No exploration",
    #              "1nn_exp3": "Yes-to-unknown exploration", "1nn_exp4": "Yes-to-percentage-unknown exploration"}
    # set_of_interest = ["Std_ext", "1nn_ext"]
    # full_name = {"Std_ext":"Epsilon-threshold assumption (1,000 rounds)", "1nn_ext": "Non-parametric assumption (1,000 rounds)"}
    # set_of_interest = ["V", "Std"]
    # full_name = {"V": "Model-based learning",
    #              "Std": "Probabilistic-based learning"}



    # theme = "validate_5a"
    # set_of_interest = ["5a_V", "5a_1nn_V","5a", "5a_1nn"]
    # full_name = {"5a_V": "5-agent; model-based; epsilon-threshold",
    #              "5a_1nn_V": "5-agent; model-based; non-parametric",
    #              "5a": "5-agent; probabilistic-based; epsilon-threshold",
    #              "5a_1nn": "5-agent; probabilistic-based; non-parametric"}
    # theme = "validate_15a"
    # set_of_interest = ["V", "1nn_V", "Std", "1nn"]
    # full_name = {"V": "15-agent; model-based; epsilon-threshold",
    #              "1nn_V": "15-agent; model-based; non-parametric",
    #              "Std": "15-agent; probabilistic-based; epsilon-threshold",
    #              "1nn": "15-agent; probabilistic-based; non-parametric"}
    # theme = "better_g95"
    # set_of_interest = ["g95"]
    # one_side_only = 0
    # full_name = {"g95": "Varying discounting rate (0.99-0.95);\nprobabilistic-based; non-parametric"}
    # theme = "learning_model"
    # set_of_interest = ["1nn_V", "1nn"]
    # full_name = {"1nn_V": "Model-based; non-parametric",
    #              "1nn": "Probabilistic-based; non-parametric"}
    # theme = "assumption"
    # set_of_interest = ["Std", "1nn"]
    # full_name = {"Std": "Epsilon-threshold; probabilistic-based",
    #              "1nn": "Non-parametric; probabilistic-based"}
    # theme = "exploration"
    # set_of_interest = ["1nn", "1nn_exp1", "1nn_exp2", "1nn_exp3", "1nn_exp4"]
    # full_name = {"1nn": "No exploration",
    #              "1nn_exp1": "Non-optimal exploration",
    #              "1nn_exp2": "Fixed Yes-No exploration",
    #              "1nn_exp3": "Yes-to-unknown exploration",
    #              "1nn_exp4": "Yes-to-percentage-unknown exploration"}
    theme = "validation_forget"
    set_of_interest = ["forget", "forget_1nn", "Std","1nn"]
    full_name = {"forget": "Forgetting; epsilon-threshold",
                 "forget_1nn": "Forgetting; non-parametric",
                 "Std": "Non-forgetting; epsilon-threshold",
                 "1nn": "Non-forgetting; non-parametric"}
    theme = "thousand"
    set_of_interest = ["Std_ext", "1nn_ext"]
    full_name = {"Std_ext": "1,000 rounds; epsilon-threshold",
                 "1nn_ext": "1,000 rounds; non-parametric"}
    number_of_players = 5
    repeat = 100
    # for i in range(21):
    # for i in [0,6,7,8,9]:
    # for i in [11,13,14,15,16]:
    # for i in [1,12]:

    learnfile = "output/results__learning_Model0.01_0.3_0.99_0_10_1_0.2_True_False_False_30_30rounds_rep0_left.tsv"
    matchfile = "output/results_matches_Model0.01_0.3_0.99_0_10_1_0.2_True_False_False_30_30rounds_rep0_left.tsv"
    for i in [19,20,21,22]:       # 5 agent
    # for i in [2,10,0,11]:       # 15 agent
    # for i in [4]:  # g95
    # for i in [10, 11]:  # learning model
    # for i in [0, 11]:  # assumption
    # for i in [11, 13, 14, 15, 16]:  # exploration
    # for i in [24,23,0,11]:  # forgetting
    # for i in [1,12]:  # thousand
        mae, converge_rate = open_pickle(i, learnfile, matchfile)
        # mae, converge_rate = open_pickle(i, set_of_interest, mae, converge_rate)
    print(list(mae.keys()))
    print(list(converge_rate.keys()))
    # open_pickle(20,["5a_nn"],mae,converge_rate)
    # open_pickle(19,["5a"],mae,converge_rate)

    plt.figure(9)
    plt.cla()
    plt.clf()
    plt.title("Mean Absolute Error (MAE)")
    X = [x for x in range(1, repeat + 1)]

    # print range(1, config.number_of_players_left+1)
    # print threshold_error_l

    for i, key in enumerate(set_of_interest):
        cmap = plt.get_cmap('jet_r')
        if one_side_only == 1:
            color = cmap(float(i) / len(set_of_interest))
            YZ = mae[key]
            plt.plot(X, YZ, label="{0}\nFinal MAE = {1}".format(full_name[key], round(YZ[-1], 2)), c=color)
        else:
            color = cmap((float(i) / len(set_of_interest))/2)
            Y = mae[key][0]
            plt.plot(X, Y, label="{0} (LEFT)\nFinal MAE = {1}".format(full_name[key], round(Y[-1], 2)), c=color)
            color = cmap((float(i) / len(set_of_interest))/2 + 0.5)
            Z = mae[key][1]
            plt.plot(X, Z, label="{0} (RIGHT)\nFinal MAE = {1}".format(full_name[key], round(Z[-1], 2)), c=color)
    # plt.legend(loc="lower left", framealpha=0.3)
    if theme == "exploration":
        plt.legend(loc="lower left", framealpha=0.6, prop={'size':11})
    else:
        plt.legend(loc="upper right", framealpha=0.3, prop={'size':11})
    plt.xlabel("Number of Round")
    plt.ylabel("Mean Absolute Error")
    plt.savefig(theme + '/' +'fig9.pdf')

    plt.figure(10)
    plt.cla()
    plt.clf()
    plt.title("Convergence Rate")
    X, Y = [x for x in range(1, repeat + 1)], []

    threshold_to_count_as_converge = 0.5

    plt.plot([0, repeat], [number_of_players, number_of_players], ls='--', c='green',
             label="Baseline for Perfect Convergence")

    for i, key in enumerate(set_of_interest):
        cmap = plt.get_cmap('jet_r')
        if one_side_only == 1:
            color = cmap(float(i) / len(set_of_interest))
            YZ = converge_rate[key]
            plt.plot(X, YZ, label="{0}\nConverge Rt. = {1}, #Unit = {2}".format(full_name[key], round(YZ[-1] / config.number_of_players, 2),YZ[-1]), c = color)
        else:
            color = cmap((float(i) / len(set_of_interest))/2)
            Y = converge_rate[key][0]
            plt.plot(X, Y, label="{0} (LEFT)\nConverge Rt. = {1}, #Unit = {2}".format(full_name[key],
                                                                                round(Y[-1] / config.number_of_players,
                                                                                      2), Y[-1]), c=color)
            color = cmap((float(i) / len(set_of_interest))/2 + 0.5)
            Z = converge_rate[key][1]
            plt.plot(X, Z, label="{0} (RIGHT)\nConverge Rt. = {1}, #Unit = {2}".format(full_name[key],
                                                                                round(Z[-1] / config.number_of_players,
                                                                                      2), Z[-1]), c=color)
            # plt.legend(loc="upper left", framealpha=0.3)
    if theme == "better_g95":
        plt.legend(loc="upper right", framealpha=0.6, prop={'size':11})
    elif theme == "exploration":
        plt.legend(loc="upper left", framealpha=0.6, prop={'size':11})
    else:
        plt.legend(loc="lower right", framealpha=0.6, prop={'size':11})
    plt.xlabel("Number of Round")
    plt.ylabel("Number of Converged Agents (Max = {0})".format(config.number_of_players))
    plt.axis([0, repeat, 0.0, config.number_of_players * 1.01])

    plt.savefig(theme + '/' +'fig10.pdf')

    plt.figure(11)
    plt.cla()
    plt.clf()
    plt.title("Mean Absolute Error (MAE)")
    X = [x for x in range(1, repeat + 1)]

    # print range(1, config.number_of_players_left+1)
    # print threshold_error_l

    for i, key in enumerate(set_of_interest):
        cmap = plt.get_cmap('jet_r')
        if one_side_only == 1:
            color = cmap(float(i) / len(set_of_interest))
            YZ = mae[key]
            plt.plot(X, YZ, label="{0}".format(full_name[key]), c=color)
        else:
            color = cmap((float(i) / len(set_of_interest)) / 2)
            Y = mae[key][0]
            plt.plot(X, Y, label="{0} (LEFT)".format(full_name[key]), c=color)
            color = cmap((float(i) / len(set_of_interest)) / 2 + 0.5)
            Z = mae[key][1]
            plt.plot(X, Z, label="{0} (RIGHT)".format(full_name[key]), c=color)
    # plt.legend(loc="lower left", framealpha=0.3)
    if theme == "exploration":
        plt.legend(loc="lower left", framealpha=0.6, prop={'size':11})
    else:
        plt.legend(loc="upper right", framealpha=0.3, prop={'size':11})
    plt.xlabel("Number of Round")
    plt.ylabel("Mean Absolute Error")
    plt.savefig(theme + '/' +'fig11.pdf')

    plt.figure(12)
    plt.cla()
    plt.clf()
    plt.title("Convergence Rate")
    X, Y = [x for x in range(1, repeat + 1)], []

    threshold_to_count_as_converge = 0.5

    plt.plot([0, repeat], [number_of_players, number_of_players], ls='--', c='green',
             label="Baseline for Perfect Convergence")

    for i, key in enumerate(set_of_interest):
        cmap = plt.get_cmap('jet_r')
        if one_side_only == 1:
            color = cmap(float(i) / len(set_of_interest))
            YZ = converge_rate[key]
            plt.plot(X, YZ,
                     label="{0}".format(full_name[key]),
                     c=color)
        else:
            color = cmap(float(i) / len(set_of_interest))
            Y = converge_rate[key][0]
            plt.plot(X, Y, label="{0} (LEFT)".format(full_name[key]), c=color)
            color = cmap((float(i) / len(set_of_interest)) / 2 + 0.5)
            Z = converge_rate[key][1]
            plt.plot(X, Z, label="{0} (RIGHT)".format(full_name[key]), c=color)
    if theme == "better_g95":
        plt.legend(loc="upper right", framealpha=0.6, prop={'size':11})
    elif theme == "exploration":
        plt.legend(loc="upper left", framealpha=0.6, prop={'size':11})
    else:
        plt.legend(loc="lower right", framealpha=0.6, prop={'size':11})
    plt.xlabel("Number of Round")
    plt.ylabel("Number of Converged Agents (Max = {0})".format(config.number_of_players))
    plt.axis([0, repeat, 0.0, config.number_of_players * 1.01])

    plt.savefig(theme + '/' +'fig12.pdf')
    plt.show()
