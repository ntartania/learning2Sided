'''
Numeber of agents in each set is given by 'count'

'''





simulation_rounds = 1 # rounds
simulation_steps = 30 # time_steps
simulation_time = 30 # dunno

# population
# player_left = {"PALO":20}
# player_right = {"PALO":20}
modelers = [{'type':"Model",
                'count':5,
                'params':{
                     # Learning rate
                     'learning_rate' :0.2 ,
                     # Discounting factor
                     'gamma' :0.99,
                     # Exploration rate
                     'epsilon' : 0.01,

                     # how to estimate yes/no probabilities
                     'method_1nn' : True,
                     'knn_param' : 10,
                     'method_epsilon_threshold' : False,
                     'method_baseline_count': False,

                     # forget the past (except for KNN approach)
                     'forgetting_factor' : 0.3, #forget this percentage of past interactions
                     'remember_min' : 30,  #(only start forgetting when there are at least this many past interactions to remember!)

                     # Log the obtained reward during learning
                     'last_episode' : 1,
                     'initial_threshold': 0 }
                 }]

modelers2 = [{'type':"Model",
                'count':5,
                'params':{
                     # Learning rate
                     'learning_rate' :0.2 ,
                     # Discounting factor
                     'gamma' :0.99,
                     # Exploration rate
                     'epsilon' : 0.01,

                     # how to estimate yes/no probabilities
                     'method_1nn' : True,
                     'knn_param' : 10,
                     'method_epsilon_threshold' : False,
                     'method_baseline_count': False,

                     # forget the past (except for KNN approach)
                     'forgetting_factor' : 0.3, #forget this percentage of past interactions
                     'remember_min' : 30,  #(only start forgetting when there are at least this many past interactions to remember!)

                     # Log the obtained reward during learning
                     'last_episode' : 1,
                     'initial_threshold': 0 }
                 }]


Qlearners2 = [{'type':"Qlearn2",
                 'count':30,
                 'params':{
                 # Learning rate
                     'alpha' :0.1 ,
                     # Discounting factor
                     'gamma' :0.99,
                     # Exploration rate
                     'epsilon' : 0.01 }
                 }]


Qlearners2explore = [{'type':"Qlearn2",
                 'count':30,
                 'params':{
                 # Learning rate
                     'alpha' :0.1 ,
                     # Discounting factor
                     'gamma' :0.99,
                     # Exploration rate
                     'epsilon' : 0.1 }
                 }]


Qlearners = [{'type':"Qlearn",
                 'count':150,
                 'params':{
                 # Learning rate
                     'alpha' :0.1 ,
                     # Discounting factor
                     'gamma' :0.99,
                     # Exploration rate
                     'epsilon' : 0.1 }
                 }]

TheoreticalAgents = [{'type':"Theoretical",
                 'count':30,
                 'params':{
                 # Learning rate
                     'agent_types' : list(range(1,31)),
                     # Discounting factor
                     'gamma' :0.99,
                 }
                 }]


player_left = modelers
player_right = modelers2



# player_left = {"best-so-far":20}
# player_right = {"best-so-far":20}
# player_left = {"random":10, "best-so-far":10, "threshold":10, "mean-so-far":10 , "secretary":10, "andria":10}
# player_right = {"random":10, "best-so-far":10, "threshold":10, "mean-so-far":10 , "secretary":10, "andria":10}
strategies = list(set([bunch['type']+ '_'.join([str(bunch['params'][p]) for p in sorted(bunch['params'].keys())]) for bunch in player_left + player_right]))
#number_of_players_left = sum(player_left.values())
#number_of_players_right = sum(player_right.values())
#number_of_players = min(number_of_players_left, number_of_players_right)

# simulation
# output_name = 'output_' + 'old.method_'
# output_name = 'output_' + '5agents_'
# output_name = 'output_'
# output_name = 'output_' + '5sims_'+ 'old.method_'
# output_name = 'output_' + '5sims_'+ '5agents_'
# output_name = 'output_' + '5sims_'+ '5agents_'
output_name = 'output/output_' + '5sims_'
old_method = 2
if old_method == 1:
    output_name += '.method_'
if old_method == 2:
    output_name += 'new.method_'
# simulation_rounds = 1
# simulation_steps = 30
output_name += 'Rep.' + str(simulation_rounds) + '_'
# output_name = 'output'+'_'+str(simulation_rounds)
# simulation_time = 30
output_name += 'Time.' + str(simulation_time) + '_'
# early_stop_epsilon = 0.01
early_stop_epsilon = 0.00
rank_analysis_mod = 1
step_update = 0
if step_update == 1:
    output_name += 'Step.updt' + '_'
method_epsilon_threshold = 1
if method_epsilon_threshold == 1:
    output_name += 'Eps.Thr' + '_'
if method_epsilon_threshold == 0:
    method_1nn = 0
    if method_1nn == 1:
        output_name += 'Mth.1nn' + '_'
# knn = 0

forgetting_factor = 0.0
if forgetting_factor > 0:
    forget = 1
    output_name += 'Forget' + '_'
else:
    forget = 0


exploration_mode = 0      # 0:no exploration , 1:non-optimal exploration, 2:fixed-yes-no exploration 3:yes-if-no-data 4:yes-if-less-than-1%
output_name += 'Explr.Md.' + str(exploration_mode) + '_'
if exploration_mode == 1:
    exploration_rate = 0.03
    output_name += 'Explr.Rt.' + str(exploration_rate) + '_'
if exploration_mode == 2:
    yes_rate = 0.015
    output_name += 'Ys.Rt.' + str(yes_rate) + '_'
    no_rate = 0.015
    output_name += 'No.Rt.' + str(no_rate) + '_'
if exploration_mode == 4:
    data_percentage = 0.01
    output_name += 'Data.Percent.' + str(data_percentage) + '_'
if exploration_mode != 0:
    shrink_exploration_rate = 0.95
    output_name += 'Explr.Rt.Shrnk.' + str(shrink_exploration_rate) + '_'
force_uniform_opponent_prob = 0
if force_uniform_opponent_prob == 1:
    output_name += 'Frce.Uniform.Prob' + '_'

plotting = 1
subjective = 0
output_name += 'Subj.' + str(subjective) + '_'
# apply_discounted_utility = 0      # 0 for no, 1 for yes
# discounting_factor = 0.9
# if apply_discounted_utility == 0:
#     discounting_factor = 1
t_0 = 1
category_of_property = subjective * 10
category_of_requirement = category_of_property
random_decimal_point = 2
replacement_mode = ["None", "Clone", "Prob-Entrance"][1]
output_name += 'Replace.' + replacement_mode + '_'
replacement_constant = (1.0 / simulation_time) * 1
if replacement_mode == "Prob-Entrance":
    output_name += 'Ent.Rt.' + str(replacement_constant) + '_'



# model_parameters
attractiveness_min = 0
attractiveness_max = 10
property_min = -5
property_max = 5
requirement_min = property_min
requirement_max = property_max
subjectivity_constant = 10
if subjective == 0:
    subjectivity_constant = 0

# references
lower_bound = attractiveness_min - (subjectivity_constant * 1)
upper_bound = attractiveness_max + (subjectivity_constant * 1)
dynamic_range = upper_bound - lower_bound
non_matched_reward = lower_bound

# # learning_parameters
# learning_rate = 0.1
# reduce_learning_rate = 1
# discount_rate = 0.99
windowing = simulation_rounds
# threshold_shrink = 0.9

# Bellman's equation parameters
Bellman_epsilon_l = 0.001
output_name += 'Bell.eps.l.' + str(Bellman_epsilon_l) + '_'
Bellman_epsilon_r = 0.001
output_name += 'Bell.eps.r.' + str(Bellman_epsilon_r) + '_'
initial_VO = 0#number_of_players/2
output_name += 'Ini.V.' + str(initial_VO) + '_'
Bellman_gamma_l = 0.99
output_name += 'Bell.gam.l.' + str(Bellman_gamma_l) + '_'
Bellman_gamma_r = 0.99
output_name += 'Bell.gam.r.' + str(Bellman_gamma_r) + '_'


# strategy_parameters initialisation
random_threshold = 0.5
# threshold = upper_bound * 0.1
# threshold = upper_bound/2
# threshold_reduction = 0.9
best_so_far = lower_bound
secretary_calibration_time = simulation_time * 0.3679
andria_calibration_time = simulation_time * 0.3679

# # PALO parameters
# PALO_threshold = threshold
# PALO_decimal_point = 1
# PALO_epsilon =20
# PALO_delta = 1
# PALO_neighbour_range = float (dynamic_range / 7)

# display
verbose_setup = 1
verbose_simulation = 0
verbose_analysis = 0
verbose_analysis_matched = 1
verbose_analysis_non_matched = 1
show_validation = 1