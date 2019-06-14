import string
import re
import random
#import Strategy
import config
import math
import Calculation
#import Create_Agents
from collections import defaultdict

class Agent(object):
    def __init__(self, ID, strategy):
        self.ID = ID
        self.attractiveness = ((ID - 1) % sum([v['count'] for v in config.player_left]) ) + 1
        # self.attractiveness = Calculation.get_uniformly_random_number(config.attractiveness_min, config.attractiveness_max, config.random_decimal_point)
        self.properties = tuple([Calculation.get_uniformly_random_number(config.property_min, config.property_max, config.random_decimal_point) for i in range(config.category_of_property)])
        self.requirements = tuple([Calculation.get_uniformly_random_number(config.requirement_min, config.requirement_max, config.random_decimal_point) for i in range(config.category_of_requirement)])
        self.strategy = strategy
        # self.opponent = None
        # self.ideal_opponent = None
        # self.ideal_opponent_left = None
        # self.ideal_opponent_right = None
        # self.rank = None
        # self.parameters = {}
        self.meeting_history = [] #list of (time, opp_type, decision, other's decision, reward)
        self.utility = []
        self.threshold_history = []

        self.time =0 # the actual learning time (counts learning events)
        self.discount_time = 0  # a timer to discount rewards over time

        # #{i+1: [] for i in range(config.simulation_rounds)}
#        self.opponent_answer = {}#{i+1: [] for i in range(config.simulation_rounds)}
#        self.reward_history = {}#{i+1: [] for i in range(config.simulation_rounds)}
        self.replacement_prob = config.replacement_constant     # may depends on time_past and self_attractiveness

    def print_instance(self):
        if not config.verbose_silentSimulations: print("  {0} {1} {2} {3} {4}".format(self.ID, self.strategy, self.attractiveness, self.properties, self.requirements))

    def print_reward(self, opponent = None):
        if not config.verbose_silentSimulations: print("  {0} {1} {2}".format(self.ID, self.strategy, self.get_reward(opponent)))

    def get_reward(self, opponent_type = None):
        if opponent_type == None:
            reward = config.non_matched_reward
        else:
            attractiveness, opp_properties = opponent_type
            # applied cosine similarity
            if config.subjective == 1:
                reward = attractiveness + config.subjectivity_constant * Calculation.cosine_similarity(self.requirements, opp_properties)
            elif config.subjective == 0:
                reward = attractiveness
        return round(reward, config.random_decimal_point)

    def get_type(self):
        return self.attractiveness, self.properties

    def get_decision(self, opponent):
        raise NotImplementedError

    def wouldmatch(self,state):
        raise NotImplementedError

    def random_decision(self):
        return int(round(random.uniform(0,1)))


    def invert_decision (self, decision):
        answer = None
        if decision == 'yes':
            answer = 'no'
        elif decision == 'no':
            answer = 'yes'
        return answer

    #def learn(self, round_update):
    #    if self.ID in Create_Agents.Left_ID:
    #        early_stop, self.parameters = Strategy.learn(self.strategy, self.parameters, self.meeting_history, self.reward_history, self.opponent_answer, round_update, config.Bellman_epsilon_l, config.Bellman_gamma_l)
    #    elif self.ID in Create_Agents.Right_ID:
            # print self.ID, "right"
            # print Create_Agents.Right_ID, self.ID
            #early_stop, self.parameters = Strategy.learn(self.strategy, self.parameters, self.meeting_history, self.reward_history, self.opponent_answer, round_update, config.Bellman_epsilon_r, config.Bellman_gamma_r)
        # if self.ID == 16:
        #     exit()
        # if self.strategy == 'threshold':
        #     print self.parameters
    #    return early_stop

    #def learn_overtime(self):
        #self.parameters = Strategy.learn_overtime(self.strategy, self.parameters, self.meeting_history)
        # if self.strategy == 'threshold':
        #     print self.parameters

    def display_secret(self, print_out = 1):
        # print self.ID
        # print Create_Agents.Left_Ranked_Attractiveness
        # print Create_Agents.Right_Ranked_Attractiveness
        if print_out:
            print(("*** secret *** attractiveness: {0}".format(self.attractiveness)))
        if self.ID in Create_Agents.Left_ID:
            rank = Create_Agents.Left_Ranked_Attractiveness.index(self.attractiveness)
            rank_order = len(Create_Agents.Left_Ranked_Attractiveness)-rank
            if print_out:
                print(("*** secret *** rank: {0} out of {1}".format(rank_order, len(Create_Agents.Left_Ranked_Attractiveness))))
            threshold_just_fit = Create_Agents.Right_Ranked_Attractiveness[rank]
            threshold_lower = "(-inf)"
            if rank - 1 >= 0:
                threshold_lower = Create_Agents.Right_Ranked_Attractiveness[rank-1]
            threshold_higher = "(inf)"
            if rank + 1 < len(Create_Agents.Right_Ranked_Attractiveness):
                threshold_higher = Create_Agents.Right_Ranked_Attractiveness[rank+1]
            if print_out:
                print(("*** secret *** suitable threshold: {0} - {1} - {2}".format(threshold_lower, threshold_just_fit, threshold_higher)))
        elif self.ID in Create_Agents.Right_ID:
            rank = Create_Agents.Right_Ranked_Attractiveness.index(self.attractiveness)
            rank_order = len(Create_Agents.Right_Ranked_Attractiveness)-rank
            if print_out:
                print(("*** secret *** rank: {0} out of {1}".format(rank_order, len(Create_Agents.Right_Ranked_Attractiveness))))
            threshold_just_fit = Create_Agents.Left_Ranked_Attractiveness[rank]
            threshold_lower = "(-inf)"
            if rank - 1 >= 0:
                threshold_lower = Create_Agents.Left_Ranked_Attractiveness[rank - 1]
            threshold_higher = "(inf)"
            if rank + 1 < len(Create_Agents.Left_Ranked_Attractiveness):
                threshold_higher = Create_Agents.Left_Ranked_Attractiveness[rank + 1]
            if print_out:
                print(("*** secret *** suitable threshold: {0} - {1} - {2}".format(threshold_lower, threshold_just_fit, threshold_higher)))
        return rank_order, threshold_lower, threshold_just_fit, threshold_higher
    def reflection(self, opponent_value):
        self.parameters = Strategy.reflection(self.strategy, self.parameters, opponent_value)

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================
#=========================================================================================================

#===================================== model-based learner

class ModelBasedAgent(Agent):
    def __init__(self, ID, strategy):
        super(ModelBasedAgent, self).__init__(ID, strategy)

        # The state is defined as the type of the opponent. there are two actions: yes (0) and no (1).
        self.Q = dict() # state (initially unknown number of states) will point to a pair where the pair represents the Q value for yes and no actions
        self.algo_type = 'Model'
        # Helper dictionaries that allow us to move from actions to
        # Q table indices and vice versa
        self.idx2act = {0: "yes", 1:"no"}
        self.act2idx = {"yes": 0, "no":1 }

        #keep track of current distribution estimates (only recalculate on new information)
        self.opp_dist = {} #=> distribution of opponent types
        self.opp_yes_prob = defaultdict(lambda:self.unseen_prob_yes) #=> probability of a yes from a given agent type

        self.prev_state = None #until we've started meeting people there's no previous state.


    def init_parameters(self, params):
        # Learning rate
        self.learning_rate = params['learning_rate']
        # Discounting factor
        self.gamma = params['gamma']
        # Exploration rate
        self.epsilon = params['epsilon']

        # how to estimate yes/no probabilities
        self.method_baseline_count = params['method_baseline_count']
        self.method_1nn = params['method_1nn']
        self.knn_param = params['knn_param']
        self.method_epsilon_threshold = params['method_epsilon_threshold']

        # forget the past (except for KNN approach)
        self.forgetting_factor = params['forgetting_factor'] #forget this percentage of past interactions
        self.remember_min = params['remember_min']  #(only start forgetting when there are at least this many past interactions to remember!)

        # Log the obtained reward during learning
        self.threshold = params['initial_threshold'] #TODO: this is the initial threshold; make this another value, maybe random

        #append parameter values to strategy identifier (useful for analysis)
        self.strategy = self.strategy + '_'.join([str(params[p]) for p in sorted(params.keys())])

        # if no data, just return (this happens if we've only said no to everyone we've met)
        self.unseen_prob_yes = params['unseen_prob_yes']
        self.picky = params['picky']
        self.pickyPercentage = params['pickyPercentage']




    #def act(self):
    def get_decision(self, opponent):

        """
        Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        score = self.get_reward(opponent.get_type())

        #if we are here it means that either (it's the start) or (prev action/state took us here with reward 0)
        # let's learn this and update our probabilities.
        if (self.prev_state): #initialized to None: will not be called on very first action of a round

            if (self.prev_action == 'yes'):
                otherdecision = 'no'
                self.learn(True, True)
            else:
                otherdecision = 'unk'
                self.learn(True, True) # no new information about yes/no probabilities, but you may need to add an unseen agent type
            self.meeting_history.append((self.time, self.prev_state, self.prev_action, otherdecision, 0))
            self.utility.append(0)

        self.time +=1
        self.discount_time +=1

        # If exploring
        #if np.random.uniform(0., 1.) < self.epsilon:
            # Select a random action using softmax
        #    Q_s = self.Q[self.state[0], self.state[1], :]
         #   probs = np.exp(Q_s) / np.sum(np.exp(Q_s))
          #  idx = np.random.choice(4, p=probs)
         #   self.action = self.idx2act[idx]
        #else:

        # Select the greedy action
        if (score > self.threshold):
            decision = "yes"
        else:
            decision = "no"
        #decision = self.idx2act[self.argmaxQsa(self.state)] #threshold decision (gets argmax)

        self.prev_state = opponent.get_type()
        self.prev_action = decision

        return decision, self.gamma **(self.discount_time) *self.get_reward(opponent.get_type())

    #simplified decision function to compute final matching sets
    def wouldmatch(self,state):
        score = self.get_reward(state)
        
        return (score>= self.threshold)

    def notify_matched(self, otherguy):

        #super(ModelBasedAgent, self).notify_matched(otherguy)
        self.meeting_history.append((self.time, self.prev_state, self.prev_action, 'yes', self.gamma **(self.discount_time) *self.get_reward(otherguy.get_type())))
        self.utility.append(self.gamma **(self.discount_time) *self.get_reward(otherguy.get_type()))
        self.learn(True, True)
        self.discount_time =0 # reset timer!!
        self.prev_state = None # forget where we were last (to avoid contaminating forward decisions)


    def learn(self, re_estimate_opp_dist, re_estimate_yes_prob): #boolean parameters: should we re-estimate the distribution of opponent types? the probability of getting yes/no answers?

        if (re_estimate_opp_dist): # calculate frequency of opponent types #TODO: estimate a continuous distribution. This could be done with a ML method if we have an expected shape (e.g. gaussian)
            for meet in self.meeting_history:
                #if meet[0] >= time * self.forgetting_factor and self.meeting_history[key] != []:      # don't forget opponent type distribution. just forget behavior
                self.opp_dist[meet[1]] = self.opp_dist.get(meet[1],0) +1

            allcount = sum(self.opp_dist.values())
            for opponent in self.opp_dist: #frequency of each opponent type
                self.opp_dist[opponent] = float(self.opp_dist[opponent]) / allcount


        if (re_estimate_yes_prob):  #estimate yes/no probabilities ===============================================================================

            if (self.method_baseline_count): #baseline counting method-------------------------------------------------
                opponent_answer_counts = self.estimate_yes_prob_counting()
                #TODO: perhaps make this a defaultdict
                opponent_answer_dist = {opp:opponent_answer_counts[opp][0]/float(opponent_answer_counts[opp][0]+opponent_answer_counts[opp][1]) for opp in opponent_answer_counts}

            elif (self.method_1nn): #k-NN
                opponent_answer_dist = self.estimate_yes_prob_knn(self.knn_param) # param is K value
            elif (self.method_epsilon_threshold): # epsilon-step distribution estimate ===============================
                opponent_answer_dist = self.estimate_yes_prob_step()
            else:
                raise KeyError("Method Error")

            # if no data, just return (this happens if we've only said no to everyone we've met)
            if len(opponent_answer_dist) == 0:
                self.opp_yes_prob = defaultdict(lambda: self.unseen_prob_yes)  # estimate prob(yes) = self.unseen_prob_yes
            else:
                self.opp_yes_prob = defaultdict(lambda: self.unseen_prob_yes, opponent_answer_dist)

            if config.verbose_simulation :
                print('opponent_answer_dist', opponent_answer_dist)

        if self.strategy == "model_solution":
            new_threshold = self.value_iteration()
        else:
            new_threshold = self.model_solution()

        self.threshold_history.append(self.threshold) # record prev threshold
        if self.picky:
            self.threshold = ( self.threshold + ( self.learning_rate * (new_threshold - self.threshold) ) ) * (1+(self.pickyPercentage/100))
        else:
            self.threshold = self.threshold + ( self.learning_rate * (new_threshold - self.threshold) )




    def value_iteration(self): #value iteration solution of the "MDP"
        sorted_opponent = sorted(self.opp_dist.keys())
        V = {}
        for opponent in sorted_opponent:
            # parameters['V'][opponent] = config.initial_VO
            V[opponent] = self.threshold
            # if opponent not in parameters['V']:
            #     parameters['V'][opponent] = parameters['threshold']      # later opponents may start with current threshold instead
        if config.verbose_simulation:
            print("V_O: {0}".format([(opponent, V[opponent]) for opponent in sorted_opponent]))

        exit_the_loop = False
        last_policy = None
        current_policy = self.threshold
        if config.verbose_simulation:
            print("*** Bellman Calculation ***")
        while not (exit_the_loop):
            # Value function update
            new_V = {}
            #print "V_O_previous: {0}".format([(opp, V_O_previous[opp]) for opp in sorted_opponent])
            for opponent in sorted_opponent:
                if opponent >= current_policy:

                    new_V[opponent] = (self.opp_yes_prob[opponent] * self.get_reward(opponent.get_type())) + ((1-self.opp_yes_prob[opponent]) * sum([self.opp_prob[opp] * self.gamma * V[opp] for opp in sorted_opponent]))

                elif opponent < current_policy:

                    new_V[opponent] = sum([self.opp_prob[opp] * self.gamma * V[opp] for opp in sorted_opponent])


            # Policy update
            if config.verbose_simulation:
                print([self.opp_prob[opp] * self.gamma * new_V[opp] for opp in sorted_opponent])

            current_policy = sum([self.opp_prob[opp] * self.gamma * new_V[opp] for opp in sorted_opponent])
            #sumprob=sum([parameters['opp_prob'][opp] for opp in sorted_opponent])
            #print "sum of probs=", sumprob
            if config.verbose_simulation:
                print("current_policy: {0}".format(current_policy))

            if last_policy != None:
                if abs(current_policy - last_policy) <= config.Bellman_epsilon_l:
                    print("*** {0} is close to {1} enough, exit the loop ***".format(current_policy, last_policy))
                    exit_the_loop = True

            last_policy = current_policy

            # # new_threshold = sum([parameters['opp_prob'][opponent] * ((parameters['opp_yes_prob'][opponent] * opponent) + ((1-parameters['opp_yes_prob'][opponent]) * config.discounting_factor * parameters['threshold'])) for opponent in parameters['opp_dist']])
            # new_threshold = sum([parameters['opp_prob'][opponent] * ((parameters['opp_yes_prob'][opponent] * self_yes_prob[opponent] * opponent) + ((1-(parameters['opp_yes_prob'][opponent] * self_yes_prob[opponent])) * config.discounting_factor * parameters['threshold'])) for opponent in parameters['opp_dist']])
        return current_policy


    def model_solution(self): #solving the MDP analytically =================================================================================
        matchable_probability = {opp: (self.opp_dist[opp] * self.opp_yes_prob[opp]) for opp in self.opp_dist if self.get_reward(opp) >= self.threshold}
        weight_average_matchable_utility = sum([self.get_reward(opp) * matchable_probability[opp] for opp in matchable_probability])
        try:
            weight_average_matchable_utility = weight_average_matchable_utility / sum(matchable_probability.values())
        except:
            pass
        r = config.Bellman_gamma_l
        p = sum(matchable_probability.values())
        u_o = weight_average_matchable_utility
        reservation_utility = r * p * u_o / (1 - r + (r * p))

        if config.verbose_simulation :
            print("reservation utility (solved)", reservation_utility)

            #print {opp: (opp_prob'][opp] , parameters['opp_yes_prob'][opp]) for opp in parameters['opp_dist'] if opp >= parameters['threshold']}
            print('sum of matchable_probability.values()', sum(matchable_probability.values()))
            print('matchable_probability', matchable_probability)
            print('weight_average_matchable_utility', weight_average_matchable_utility)

            print('r', r)
            print('p', p)
            print('u_o', u_o)

        return reservation_utility

    '''
    estimate yes/no probabilities by just counting past interactions with people
    '''
    def estimate_yes_prob_counting(self):
        opponent_answer_dist = {}
        if not self.meeting_history:
            return opponent_answer_dist
        nowtime = self.meeting_history[-1][0]
        remember_time = min (round(nowtime * (1.0-self.forgetting_factor)), self.remember_min) #e.g. I remember 1.0-0.3 = 70% of past interactions count, but this value should be at least 30

        past_remembered = self.meeting_history[max(0, nowtime-remember_time):]
        for (t, otype, dec, odecision, rew) in [record for record in past_remembered if record[3]!='unk']:
            if otype not in opponent_answer_dist:
                opponent_answer_dist[otype] = [0, 0]

            if odecision == 'yes':
                opponent_answer_dist[otype][0] += 1
            elif odecision == 'no':
                opponent_answer_dist[otype][1] += 1
            else:#should have been filtered out
                raise ValueError
                exit()

        return opponent_answer_dist


    '''
    estimate probability distribution for answers yes/no conditional to opp type, using knn method
    '''
    def estimate_yes_prob_knn(self, k):
        opp_types = set ([ot for (t, ot, dec, odec, rew) in self.meeting_history])
        opp_yes_prob = {}
        for opponent in opp_types: # consider use all known opponent types
            #1) only get known yes/no answers (filter)
            #2) sort list by (a) first member of opponent types (attractiveness) / difference to considered type (KLUDGE: actually need to compare types more generally)
            #                (b) time (*-1, so that ascending order gives us most recent first)
            # truncate list to keep k

            YNdata = [t_ot_dec_odec_rew for t_ot_dec_odec_rew in self.meeting_history if t_ot_dec_odec_rew[3] != 'unk']
            #print 'len(YNdata)', YNdata, opponent
            YNdata = sorted(YNdata, key= lambda meet: (abs(meet[1][0]-opponent[0]), meet[0]*-1))
            YNdata = YNdata[0:k] # use first k: closest and most recent among closest

            if (len(YNdata)>0):
                countyes = len([odec for (t, ot, dec, odec, rew) in YNdata if odec == 'yes'])
                countno = len([odec for (t, ot, dec, odec, rew) in YNdata if odec == 'no'])

                opp_yes_prob[opponent] = float(countyes) / (countyes + countno)
            else:
                opp_yes_prob[opponent] = 0 #TODO: this will only happen at round 0000; need to initialize this somehow

        if config.verbose_simulation:
            print('KNN opp_yes_prob[opponent]', opp_yes_prob)
        return opp_yes_prob


    def estimate_yes_prob_step(self): #======================================== estimate conditional distribution of yes/no answers using step model
        opp_types = set ([ot for (t, ot, dec, odec, rew) in self.meeting_history])
        opp_yes_prob = {}

        counting_answer_dist = self.estimate_yes_prob_counting() # get the basic counts; this includes forgetting the past

        #if no data, just return (this happens if we've only said no to everyone we've met)
        if len(counting_answer_dist) == 0:
            return defaultdict(lambda:self.unseen_prob_yes) # estimate prob(yes) = self.unseen_prob_yes

        min_epsilon_threshold = max([key[0]+0.00001 for key in opp_types])
        min_epsilon_threshold_utility = float(sum([counting_answer_dist[opponent][1] for opponent in counting_answer_dist])) / \
                                                  sum([counting_answer_dist[opponent][0] + counting_answer_dist[opponent][1] for opponent in counting_answer_dist])

        # print min_epsilon_threshold_utility
        for key in counting_answer_dist:
            potential_epsilon = float(sum([counting_answer_dist[opponent][0] for opponent in counting_answer_dist if opponent >= key]) + sum([counting_answer_dist[opponent][1] for opponent in counting_answer_dist if opponent < key])) \
                                    / sum([counting_answer_dist[opponent][0] + counting_answer_dist[opponent][1] for opponent in counting_answer_dist])

                # print potential_epsilon
            if potential_epsilon < min_epsilon_threshold_utility:
                min_epsilon_threshold_utility = potential_epsilon
                min_epsilon_threshold = key[0]

        # print('min_epsilon_threshold', min_epsilon_threshold)
        # print('min_epsilon_threshold_utility', min_epsilon_threshold_utility)

        for opponent in opp_types:
            oppRew = self.get_reward(opponent)
            # print(oppRew, "  ", min_epsilon_threshold)
            if  oppRew>= min_epsilon_threshold:
                opp_yes_prob[opponent] = min_epsilon_threshold_utility
            elif oppRew < min_epsilon_threshold:
                opp_yes_prob[opponent] = 1.0 - min_epsilon_threshold_utility
        return opp_yes_prob

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================
#=========================================================================================================

class QAgent(Agent):
    def __init__(self, ID, strategy):
        super(QAgent, self).__init__(ID, strategy)

        # The state is defined as the type of the opponent. there are two actions: yes (0) and no (1).
        self.Q = dict() # state (initially unknown number of states) will point to a pair where the pair represents the Q value for yes and no actions
        self.algo_type = 'Qlearn'
        # Helper dictionaries that allow us to move from actions to
        # Q table indices and vice versa
        self.idx2act = {0: "yes", 1:"no"}
        self.act2idx = {"yes": 0, "no":1 }
        self.prev_state = None #until we've started meeting people there's no previous state.

    def init_parameters(self, params):
        # Learning rate TODO: get these from the config file
        self.alpha = params['alpha']
        # Discounting factor
        self.gamma = params['gamma']
        # Exploration rate
        self.epsilon = params['epsilon']
        self.picky = params['picky']
        self.pickyPercentage = params['pickyPercentage']

        #append parameter values to strategy identifier (useful for analysis)
        self.strategy = self.strategy + '_'.join([str(params[p]) for p in sorted(params.keys())])


    #def act(self):
    def get_decision(self, opponent):
        """
        Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        score = self.get_reward(opponent.get_type())
        #if we are here it means that either (it's the start) or (prev action/state took us here with reward 0)

        if (self.prev_state): #initialized to None: will not be called on very first action of a round
            self.learn(self.prev_state, self.prev_action, 0, opponent.get_type()) #
            if (self.prev_action == 'yes'):
                otherdecision = 'no'
            else:
                otherdecision = 'unk'
            self.meeting_history.append((self.time, self.prev_state, self.prev_action, otherdecision, 0))
            self.utility.append(0)

        self.time +=1
        self.discount_time +=1

        # If exploring
        if random.uniform(0., 1.) < self.epsilon:
            decision = self.idx2act[self.random_decision()]
        else:
            #if config.verbose_simulation:
            #print "greedy decision:", opponent.get_type(),"->", self.Q.get(opponent.get_type(),'[no data]')
            decision = self.idx2act[self.argmaxQsa(opponent.get_type())] #threshold decision (gets argmax)

        # Select the greedy action

        self.prev_state = opponent.get_type()
        self.prev_action = decision

        return decision, self.gamma ** self.discount_time * score


    def notify_matched(self, otherguy):
        #super(QAgent, self).notify_matched(otherguy)
        self.learn(self.prev_state, self.prev_action, self.get_reward(otherguy.get_type()), None)
        self.meeting_history.append((self.time, self.prev_state, self.prev_action, 'yes', self.gamma ** self.discount_time *self.get_reward(otherguy.get_type())))
        self.utility.append(self.gamma ** self.discount_time *self.get_reward(otherguy.get_type()))
        self.discount_time =0 # reset timer!!
        self.prev_state = None # forget where we were last (to avoid contaminating forward decisions)



    def learn(self, prev_state, prev_action, reward, new_state):
        # Read the current state-action value
        if (prev_state in self.Q ) and (self.act2idx[prev_action] in self.Q[prev_state]):
            Q_sa = self.Q[prev_state][self.act2idx[prev_action]]

            # Calculate the updated state action value
            if (new_state): #(not matched)
                Q_sa_new = Q_sa + self.alpha * ( self.gamma * self.maxQsa(new_state) - Q_sa) #reward is 0
            else:
                Q_sa_new = Q_sa + self.alpha * (reward - Q_sa) #new state is terminal state
        else:
            if (prev_state not in self.Q ):
                self.Q[prev_state] = {} #put the state in Q
            # Use this as initial state action value
            if (new_state): #(not matched)
                Q_sa_new = self.gamma * self.maxQsa(new_state)
            else:
                Q_sa_new = reward #new state is terminal state

        # Write the updated value

        self.Q[prev_state][self.act2idx[prev_action]] = Q_sa_new
        self.update_threshold_history()

    def update_threshold_history(self):
        ########### record history, whether this is a threshold policy and its approximate value
        is_threshold_policy = True
        below_threshold = True
        prev_t = 0
        threshold = None
        for s in sorted(list(self.Q.keys()), key=lambda k: self.get_reward(k)):
            if (below_threshold and self.Q[s].get(0,0) > self.Q.get(1,0)): #* (self.Q[get_reward(s) - self.Q_S_NO) <0: #opp_reward is more than QSNO and QS[opp] is less than QSNO, or vice-versa: our policy is not a good threshold strategy
                below_threshold = False
                threshold = 0.5 *(prev_t + self.get_reward(s))
            elif (not below_threshold and self.Q[s].get(0,0) < self.Q.get(1,0)):
                is_threshold_policy = False
            prev_t = self.get_reward(s)
        if (below_threshold): #if we never switch to above a threshold, there's no threshold
            is_threshold_policy = False
        if (not is_threshold_policy):
            threshold = None
        self.threshold_history.append(threshold)

    def maxQsa(self, state):
        if (state not in self.Q):
            return 0
        return max(self.Q[state].get(0, 0), self.Q[state].get(1, 0)) #max reward between saying yes or no in a given state, default 0 if unseen

    def wouldmatch(self,state):
        if (state not in self.Q):
            return False
            #return  Q(S,'yes') > Q(S,'no')
        return (self.Q[state][self.act2idx['yes']]>= self.Q[state][self.act2idx['no']])

    def argmaxQsa(self, state):
        if state in self.Q:
            if (len(self.Q[state])==1): #there's just one known action
                return list(self.Q[state].keys())[0] #return whatever action that is
            #else:
            if self.Q[state][0] > self.Q[state][1]:
                return 0
            elif self.Q[state][0] < self.Q[state][1]:
                return 1
        return self.random_decision()
        #return np.argmax(self.Q[state[0], state[1], :])


#=========================================================================================================

class QAgent2(QAgent): #this one does Q learning but understands that the expected value of a "no" is the same regardless of which state the agent starts from.
    def __init__(self, ID, strategy):
        super(QAgent2, self).__init__(ID, strategy)
        self.Q_S_NO = 0 # single variable to learn the expected value of saying no
        self.algo_type = 'Qlearn2'
        self.threshold_history = []

    #def act(self):
    def get_decision(self, opponent):
        """
        Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        score = self.get_reward(opponent.get_type())
        #if we are here it means that either (it's the start) or (prev action/state took us here with reward 0)

        if (self.prev_state): #initialized to None: will not be called on very first action of a round
            self.learn(self.prev_state, self.prev_action, 0, opponent.get_type()) #
            if (self.prev_action == 'yes'):
                otherdecision = 'no'
            else:
                otherdecision = 'unk'
            self.meeting_history.append((self.time, self.prev_state, self.prev_action, otherdecision, 0))
            self.utility.append(0)

        self.time +=1
        self.discount_time +=1

        # If exploring
        if random.uniform(0., 1.) < self.epsilon:
            decision = self.idx2act[self.random_decision()]
        else:
            #print "Q2 decision", opponent.get_type(), self.Q.get(opponent.get_type(), '[no data]'), self.Q_S_NO
            decision = self.idx2act[self.argmaxQsa(opponent.get_type())] #threshold decision (gets argmax)

        # Select the greedy action

        self.prev_state = opponent.get_type()
        self.prev_action = decision

        return decision, self.gamma ** self.discount_time * score


    def notify_matched(self, otherguy):
        #super(QAgent, self).notify_matched(otherguy)
        self.learn(self.prev_state, self.prev_action, self.get_reward(otherguy.get_type()), None)
        self.meeting_history.append((self.time, self.prev_state, 'yes', 'yes', self.gamma ** self.discount_time *self.get_reward(otherguy.get_type())))
        self.utility.append(self.gamma ** self.discount_time *self.get_reward(otherguy.get_type()))
        self.discount_time =0 # reset timer!!
        self.prev_state = None # forget where we were last (to avoid contaminating future decisions)

    def learn(self, prev_state, prev_action, reward, new_state): ####### Q-learning, adapted

        if prev_action == 'no':
            #print 'updating after....', prev_state, prev_action, reward, new_state
            Q_sa = self.Q_S_NO
            Q_sa_new = Q_sa + self.alpha * ( self.gamma * self.maxQsa(new_state) - Q_sa)
            self.Q_S_NO = Q_sa_new
        else:
        # Read the current state-action value
            if (prev_state in self.Q ):
                Q_sa = self.Q[prev_state]

                # Calculate the updated state action value
                if (new_state): #(not matched)
                    Q_sa_new = Q_sa + self.alpha * ( self.gamma * self.maxQsa(new_state) - Q_sa) #reward is 0
                else:
                    Q_sa_new = Q_sa + self.alpha * (reward - Q_sa) #new state is terminal state
            else:
                # Use this as initial state action value
                if (new_state): #(not matched)
                    Q_sa_new = self.gamma * self.maxQsa(new_state)
                else:
                    Q_sa_new = reward #new state is terminal state

        # Write the updated value

            self.Q[prev_state] = Q_sa_new
        if self.picky:
            # print(self.Q_S_NO)
            self.Q_S_NO = self.Q_S_NO * (1 + (self.pickyPercentage / 100))
        # print(self.Q_S_NO)
        # print("____________________")
        is_threshold_policy = True
        for s in self.Q:
            if (self.Q[s] - self.Q_S_NO) * (self.get_reward(s) - self.Q_S_NO) <0: #opp_reward is more than QSNO and QS[opp] is less than QSNO, or vice-versa: our policy is not a good threshold strategy
                is_threshold_policy = False

        self.threshold_history.append((is_threshold_policy, self.Q_S_NO)) # record whether it's the right shape, the value.

    def wouldmatch(self,state):
        if (state not in self.Q):
            return False
        return self.Q[state]>= self.Q_S_NO

    def maxQsa(self, state):
        if (state not in self.Q):
            return self.Q_S_NO
        return max(self.Q[state], self.Q_S_NO) #max reward between saying yes or no in a given state, default 0 if unseen

    def argmaxQsa(self, state):
        if state in self.Q:
            if self.Q[state] > self.Q_S_NO:
                return 0
            elif self.Q_S_NO< self.Q[state]:
                return 1
        return self.random_decision()
        #return np.argmax(self.Q[state[0], state[1], :])


#=========================================================================================================

class WolfAgent(QAgent): #this one does Q learning but understands that the expected value of a "no" is the same regardless of which state the agent starts from.
    def __init__(self, ID, strategy):
        super(WolfAgent, self).__init__(ID, strategy)
        self.Q_S_NO = 0 # single variable to learn the expected value of saying no
        self.piSA= dict()    #maintains a probability distribution to decide which action to take; action is not systematically greedy but stochastic
                             # since actions are just yes/no, maintains pi(S, yes) for all states
        self.C = dict() #C(s) count of how many times we've encountered s (used to determine average policy)
        self.pibarSA = dict() #pi bar average historical policy (for yes action)
        self.algo_type = 'Wolf'
        self.threshold_history = []
        #self.prev_state = None #until we've started meeting people there's no previous state.
        # Helper dictionaries that allow us to move from actions to
        # Q table indices and vice versa
        #self.idx2act = {0: "yes", 1:"no"}
        #self.act2idx = {"yes": 0, "no":1 }


    def init_parameters(self, params):
        # Learning rate TODO: get these from the config file
        self.alpha = params['alpha']
        #other learning rates
        self.delta_win = params['delta_win']        
        #other learning rate
        self.delta_lose = params['delta_lose']        #note: delta_win must be < delta_lose

        #exploratin rate
        self.epsilon = params['epsilon']
        # Discounting factor
        self.gamma = params['gamma']

    #def act(self):
    def get_decision(self, opponent):
        """
        Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        this_state = opponent.get_type()
        score = self.get_reward(this_state)
        #print 'facing:', score
        #if we are here it means that either (it's the start) or (prev action/state took us here with reward 0)

        if (self.prev_state): #initialized to None: will not be called on very first action of a round
            self.learn(self.prev_state, self.prev_action, 0, opponent.get_type()) #
            if (self.prev_action == 'yes'):
                otherdecision = 'no'
            else:
                otherdecision = 'unk'
            self.meeting_history.append((self.time, self.prev_state, self.prev_action, otherdecision, 0))
            self.utility.append(0)

        self.time +=1
        self.discount_time +=1

        # If exploring
        if random.uniform(0., 1.) < self.epsilon:
            decision = self.idx2act[self.random_decision()]
            #print 'exploring -', decision

        else:
            #TODO: put in mixed policy here: select an action according to the probabilities pi(s,a)
            #print "Q2 decision", opponent.get_type(), self.Q.get(opponent.get_type(), '[no data]'), self.Q_S_NO
            #stochastic decision based on piSA
            yes_prob = self.piSA.setdefault(this_state, 0.5) # get pi(S,yes) for S, using 0.5 (and setting it in dict) if the state was unseen.
            if (random.uniform(0., 1.) < yes_prob):
                decision = 'yes' #threshold decision (gets argmax)
            else:
                decision = 'no'
            #print 'NOT exploring ->', decision, '(', yes_prob,')'

        self.prev_state = opponent.get_type()
        self.prev_action = decision

        return decision, self.gamma ** self.discount_time * score


    def notify_matched(self, otherguy):
        #super(QAgent, self).notify_matched(otherguy)
        self.learn(self.prev_state, self.prev_action, self.get_reward(otherguy.get_type()), None)
        self.meeting_history.append((self.time, self.prev_state, 'yes', 'yes', self.gamma ** self.discount_time *self.get_reward(otherguy.get_type())))
        self.utility.append(self.gamma ** self.discount_time *self.get_reward(otherguy.get_type()))
        self.discount_time =0 # reset timer!!
        self.prev_state = None # forget where we were last (to avoid contaminating future decisions)

    def learn(self, prev_state, prev_action, reward, new_state): ####### Q-learning, adapted

        
        if prev_action == 'no':
            #print 'updating after....', prev_state, prev_action, reward, new_state
            Q_sa = self.Q_S_NO
            Q_sa_new = Q_sa + self.alpha * ( self.gamma * self.maxQsa(new_state) - Q_sa)
            self.Q_S_NO = Q_sa_new

        else:
        # Read the current state-action value
            if (prev_state in self.Q ):
                Q_sa = self.Q[prev_state]

                # Calculate the updated state action value
                if (new_state): #(not matched)
                    Q_sa_new = Q_sa + self.alpha * ( self.gamma * self.maxQsa(new_state) - Q_sa) #reward is 0
                else:
                    Q_sa_new = Q_sa + self.alpha * (reward - Q_sa) #new state is terminal state
                    #print 'match with',reward,' QSA',Q_sa_new,' Q_S_NO',self.Q_S_NO, ' me:', self.ID
            else:
                # Use this as initial state action value
                if (new_state): #(not matched)
                    Q_sa_new = self.gamma * self.maxQsa(new_state)
                else:
                    Q_sa_new = reward #new state is terminal state
                    #print 'match with',reward,' QSA',Q_sa_new,' Q_S_NO',self.Q_S_NO, ' me:', self.ID

        # Write the updated value

            self.Q[prev_state] = Q_sa_new
        
        
        #update C(s) then avg policy
        self.C[prev_state] = self.C.setdefault(prev_state,0) + 1
        
        #Update average policy pibarSA
        current_pibar = self.pibarSA.setdefault(prev_state,0.5) # initial value for pibar is 0.5, same as for pi
        self.pibarSA[prev_state]= current_pibar + (1.0/self.C[prev_state])*(self.piSA.setdefault(prev_state, 0.5) - current_pibar)

        #distinguish "winning" and "losing"
        if ((self.piSA[prev_state]*self.Q.get(prev_state,0)+(1-self.piSA[prev_state])*self.Q_S_NO)>
            (self.pibarSA[prev_state]*self.Q.get(prev_state,0)+(1-self.pibarSA[prev_state])*self.Q_S_NO)):
            winning = True
        else:
            winning = False

        #update pi_S_A as a mixed policy
        if (winning):
            delta= self.delta_win
        else:
            delta = self.delta_lose

        if (self.Q_S_NO>self.Q.get(prev_state,0)):
            self.piSA[prev_state] = max(0, self.piSA[prev_state] - delta)
        elif (self.Q_S_NO<self.Q.get(prev_state,0)):
            self.piSA[prev_state] = min(1, self.piSA[prev_state] + delta)

        
        # print(self.Q_S_NO)
        # print("____________________")
        is_threshold_policy = True
        for s in self.Q:
            if (self.Q[s] - self.Q_S_NO) * (self.get_reward(s) - self.Q_S_NO) <0: #opp_reward is more than QSNO and QS[opp] is less than QSNO, or vice-versa: our policy is not a good threshold strategy
                is_threshold_policy = False

        self.threshold_history.append((is_threshold_policy, self.Q_S_NO)) # record whether it's the right shape, the value.

    def maxQsa(self, state):
        if (state not in self.Q):
            return self.Q_S_NO
        return max(self.Q[state], self.Q_S_NO) #max reward between saying yes or no in a given state, default 0 if unseen

    def wouldmatch(self,state):
        if (state not in self.Q):
            return False
        return self.piSA[state]> 0.5

    def argmaxQsa(self, state):
        if state in self.Q:
            if self.Q[state] > self.Q_S_NO:
                return 0
            elif self.Q_S_NO< self.Q[state]:
                return 1
        return self.random_decision()
        #return np.argmax(self.Q[state[0], state[1], :])




#===================================================================#===================================================================
#================================================== Theoretical agent  =================================================================
#===================================================================#===================================================================

class Theoragent(Agent):

    def __init__(self, ID, strategy):
        super(Theoragent, self).__init__(ID, strategy)

        # The state is defined as the type of the opponent. there are two actions: yes (0) and no (1).
        self.Q = dict() # state (initially unknown number of states) will point to a pair where the pair represents the Q value for yes and no actions
        self.algo_type = 'Theoretical'
        # Helper dictionaries that allow us to move from actions to
        # Q table indices and vice versa
        self.idx2act = {0: "yes", 1:"no"}
        self.act2idx = {"yes": 0, "no":1 }

        #keep track of current distribution estimates (only recalculate on new information)
        self.opp_dist = {} #=> distribution of opponent types
        self.opp_yes_prob = defaultdict(lambda:self.unseen_prob_yes)  #=> probability of a yes from a given agent type

        self.prev_state = None #until we've started meeting people there's no previous state.


    def init_parameters(self, params):
        # Learning rate #TODO: get these from the config file
        self.gamma = params['gamma']

        agent_types = list(range(params['agent_types_from'],params['agent_types_to']))  #(only start forgetting when there are at least this many past interactions to remember!)

        # Log the obtained reward during learning
        calc_obj = Calculation.Calculation()
        allthresholds = calc_obj.calculate_theoretical_result(agent_types, self.gamma)
        self.threshold = allthresholds[self.get_reward(self.get_type())]

        #append paameter values to strategy identifier (useful for analysis)
        self.strategy = self.strategy + '_' + str(self.gamma) #.join([str(params[p]) for p in sorted(params.keys())])

        # if no data, just return (this happens if we've only said no to everyone we've met)
        self.unseen_prob_yes = params['unseen_prob_yes']

    #def act(self):
    def get_decision(self, opponent):

        """
        Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        score = self.get_reward(opponent.get_type())

        if (score >self.threshold):
            decision = "yes"
        else:
            decision = "no"
        #decision = self.idx2act[self.argmaxQsa(self.state)] #threshold decision (gets argmax)

        self.prev_state = opponent.get_type()
        self.prev_action = decision

        return decision, self.gamma **(self.discount_time) *self.get_reward(opponent.get_type())

    #simplified decision function to compute final matching sets
    def wouldmatch(self,state):
        score = self.get_reward(state)
        return (score>= self.threshold)


    def notify_matched(self, otherguy):

        #super(ModelBasedAgent, self).notify_matched(otherguy)
        self.meeting_history.append((self.time, self.prev_state, self.prev_action, 'yes', self.gamma **(self.discount_time) *self.get_reward(otherguy.get_type())))
        self.utility.append(self.gamma **(self.discount_time) *self.get_reward(otherguy.get_type()))
        self.discount_time = 0 # reset timer!!
        self.prev_state = None # forget where we were last (to avoid contaminating forward decisions)







#===================================================================#===================================================================
#================================================== RANDOM agent  =================================================================
#===================================================================#===================================================================

class Random(Agent):

    def __init__(self, ID, strategy):
        super(Random, self).__init__(ID, strategy)

        # The state is defined as the type of the opponent. there are two actions: yes (0) and no (1).
        self.Q = dict() # state (initially unknown number of states) will point to a pair where the pair represents the Q value for yes and no actions
        self.algo_type = 'Random'
        # Helper dictionaries that allow us to move from actions to
        # Q table indices and vice versa
        self.idx2act = {0: "yes", 1:"no"}
        self.act2idx = {"yes": 0, "no":1 }

        #keep track of current distribution estimates (only recalculate on new information)
        self.opp_dist = {} #=> distribution of opponent types
        self.opp_yes_prob = defaultdict(lambda:self.unseen_prob_yes)  #=> probability of a yes from a given agent type

        self.prev_state = None #until we've started meeting people there's no previous state.


    def init_parameters(self, params):
        # Learning rate #TODO: get these from the config file
        self.gamma = params['gamma']

        allthresholds={}
        for i in range(params['agent_types_from'],params['agent_types_to']):
            allthresholds[i] = random.uniform(1, 20)
        self.threshold = allthresholds[self.get_reward(self.get_type())]

        #append parameter values to strategy identifier (useful for analysis)
        self.strategy = self.strategy + '_' + str(self.gamma) #.join([str(params[p]) for p in sorted(params.keys())])

        # if no data, just return (this happens if we've only said no to everyone we've met)
        self.unseen_prob_yes = params['unseen_prob_yes']

    #def act(self):
    def get_decision(self, opponent):

        """
        Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        score = self.get_reward(opponent.get_type())

        if (score >self.threshold):
            decision = "yes"
        else:
            decision = "no"
        #decision = self.idx2act[self.argmaxQsa(self.state)] #threshold decision (gets argmax)

        self.prev_state = opponent.get_type()
        self.prev_action = decision

        return decision, self.gamma **(self.discount_time) *self.get_reward(opponent.get_type())

    #simplified decision function to compute final matching sets
    def wouldmatch(self,state):
        score = self.get_reward(state)
        return (score>= self.threshold)



    def notify_matched(self, otherguy):

        #super(ModelBasedAgent, self).notify_matched(otherguy)
        self.meeting_history.append((self.time, self.prev_state, self.prev_action, 'yes', self.gamma **(self.discount_time) *self.get_reward(otherguy.get_type())))
        self.utility.append(self.gamma **(self.discount_time) *self.get_reward(otherguy.get_type()))
        self.discount_time = 0 # reset timer!!
        self.prev_state = None # forget where we were last (to avoid contaminating forward decisions)