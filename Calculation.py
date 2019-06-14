import math
import random
import numpy as np
import copy


def cosine_similarity(vector_a, vector_b):
    return (sum([a * b for a, b in zip(vector_a, vector_b)]) / (
    math.sqrt(sum([a ** 2 for a in vector_a])) * math.sqrt(sum([b ** 2 for b in vector_b]))))


def get_uniformly_random_number(mininum, maximum, floating_point):
    return round(random.uniform(mininum, maximum), floating_point)


def pearson_correlation_coefficient(vector_x, vector_y):
    x_bar = np.average(vector_x)
    y_bar = np.average(vector_y)
    return (sum([(x_i - x_bar) * (y_i - y_bar) for x_i, y_i in zip(vector_x, vector_y)])) / (
    math.sqrt(sum([(x_i - x_bar) ** 2 for x_i in vector_x])) * math.sqrt(sum([(y_i - y_bar) ** 2 for y_i in vector_y])))


class Calculation:

    def __init__(self):
        self.theo = None

    def calculate_theoretical_result(self, utility_list, gamma):
        if (self.theo):
            return theo
        else:
            no_players = len(utility_list)
            r_l = gamma
            player_l = {}
            threshold_l = {}
            threshold_l_prev = {}
            sorted_rank = [i+1 for i in range(no_players)]

        # initialise selected list
            selected_for_l = {}
            selected_and_matchable_for_l = {}
            for i in sorted_rank:
                selected_for_l[i] = [j+1 for j in range(i)]

            desc_values = sorted(utility_list,reverse=True) #descending utilities
            for i in sorted_rank:
                player_l[i] = desc_values[i-1]

            has_changes = True #stopping criterion
            cnt =0
            while (has_changes):
                cnt +=1
                #print cnt , 'iterations'
                for l_rank in sorted_rank:
                    # print "loop start", l_rank
                    E_U_max = 0
                    for aim_for in sorted_rank:
                        selected_and_matchable = []
                        u_of_selected_and_matchable = {}
                        for runner in sorted_rank:
                            if runner <= aim_for and l_rank in selected_for_l[runner]:
                                selected_and_matchable.append(runner)
                    # if l_rank == 5:
                    #     print selected_and_matchable
                        if selected_and_matchable:
                            for agent in selected_and_matchable:
                                u_of_selected_and_matchable[agent] = player_l[agent]
                            u_O = np.average(list(u_of_selected_and_matchable.values()))
                            p = float(len(u_of_selected_and_matchable))/no_players
                            E_U = ( r_l * p * u_O )/( 1 - r_l + ( r_l * p ))
                            if E_U > E_U_max:
                                E_U_max = E_U
                                selected_and_matchable_for_l[l_rank] = selected_and_matchable
                    # print E_U_max
                    threshold_l[l_rank] = E_U_max
                    # print threshold_l


                for i in sorted_rank:
                    selected_for_l[i] = [rank for rank in sorted_rank if player_l[rank] >= threshold_l[i]]

                #print threshold_l
                #print threshold_r

                #print selected_and_matchable_for_l
                #print selected_and_matchable_for_r


            # stopping condition

                if threshold_l_prev:
                    has_changes = False
                    for key in threshold_l_prev:
                        if threshold_l_prev[key] != threshold_l[key]:
                            has_changes = True
                            break

                threshold_l_prev = copy.deepcopy(threshold_l)

        self.theo = {player_l[r]:threshold_l[r] for r in sorted_rank}
        return self.theo

    def test(self):
        calc = self.calculate_theoretical_result([3*i for i in range(1,18)], 0.95)
        print('1-8 results=', calc)
