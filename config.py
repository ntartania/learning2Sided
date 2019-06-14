import ast
import sys, getopt

def read_args(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> //expected string with file name and extension')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        # elif opt in ("-o", "--ofile"):
        #     outputfile = arg
    # print('Input file is: ', inputfile)
    return inputfile


config_file = str(read_args(sys.argv[1:]))
if len(config_file) == 0:
    config_file = "quick_test_parameters.config"
    # print("\n***********************\nCould not read option -i <inputfile>. \nPlease specify input file from command line\n***********************\n")
    # sys.exit(2)

with open(config_file) as file:
    parameters = {'output_notes':""}
    for line in file:# skip_empty_lines(file):
            if line[0] == '#':
                # skip any lines in the config file which starts with #
                continue
            try:
                (key, val) = line.rstrip().split(": ") # The rstrip method gets rid of the "\n" at the end of each line
                parameters[key] = val
            except:
                pass

if parameters['output_notes'] == "":
    output_notes = parameters['output_notes']
    #sim_res_pickle_name = parameters['output_notes'] + " | " + "r" + parameters['simulation_rounds'] + "s" + parameters['simulation_steps'] + ".tsv"
    sim_res_pickle_name = parameters['algo_left'] + " vs " + parameters['algo_right'] + " | " + "r" + parameters['simulation_rounds'] + "s" + parameters['simulation_steps'] + ".tsv"
    # sim_res_pickle_name = "Left " + parameters['player_left'] + " " + parameters['algo_left'] + " | Right " + parameters['player_right'] + " " + parameters['algo_right'] + " | Rounds " + parameters['simulation_rounds'] + " | Steps " + parameters['simulation_steps'] + ".pkl"
else:
    output_notes = parameters['output_notes']
    sim_res_pickle_name = parameters['output_notes'] + " | " + "r" + parameters['simulation_rounds'] + "s" + parameters['simulation_steps'] + ".tsv"
    # sim_res_pickle_name = "Left " + parameters['player_left'] + " "+parameters['algo_left'] + " | Right " + parameters['player_right'] + " "+parameters['algo_right']+" | Rounds "+ parameters['simulation_rounds'] +" | Steps "+ parameters['simulation_steps'] +" | "+parameters['output_notes'] +".pkl"

simulation_rounds = int(parameters['simulation_rounds'])
simulation_steps = int(parameters['simulation_steps'])
simulation_time = 30 # dunno

modelers_L = [{'type':"Model", 'params':ast.literal_eval(parameters['modelers_L_params']) }]
modelers_R = [{'type':"Model", 'params':ast.literal_eval(parameters['modelers_R_params']) }]

WolfAgents_L = [{'type':"Wolf", 'params':ast.literal_eval(parameters['WolfAgents_L_params']) }]
WolfAgents_R = [{'type':"Wolf", 'params':ast.literal_eval(parameters['WolfAgents_R_params']) }]

Qlearners_L = [{'type':"Qlearn", 'params':ast.literal_eval(parameters['Qlearners_L_params']) }]
Qlearners_R = [{'type':"Qlearn", 'params':ast.literal_eval(parameters['Qlearners_R_params']) }]

Qlearners2_L = [{'type':"Qlearn2", 'params':ast.literal_eval(parameters['Qlearners2_L_params']) }]
Qlearners2_R = [{'type':"Qlearn2", 'params':ast.literal_eval(parameters['Qlearners2_R_params']) }]

Qlearners2Exp_L = [{'type':"Qlearn2", 'params':ast.literal_eval(parameters['Qlearners2Exp_L_params']) }]
Qlearners2Exp_R = [{'type':"Qlearn2", 'params':ast.literal_eval(parameters['Qlearners2Exp_R_params']) }]

TheoreticalAgents_L = [{'type':"Theoretical", 'params':ast.literal_eval(parameters['TheoreticalAgents_L_params']) }]
TheoreticalAgents_R = [{'type':"Theoretical", 'params':ast.literal_eval(parameters['TheoreticalAgents_R_params']) }]

Random_L = [{'type':"Random", 'params':ast.literal_eval(parameters['Random_L_params']) }]
Random_R = [{'type':"Random", 'params':ast.literal_eval(parameters['Random_R_params']) }]


possible_algo = {'modelers_L':modelers_L, 'modelers_R':modelers_R, 'Qlearners_L':Qlearners_L, 'Qlearners_R':Qlearners_R, 'Qlearners2_L':Qlearners2_L, 'Qlearners2_R':Qlearners2_R, 'Qlearners2Exp_L':Qlearners2Exp_L, 'Qlearners2Exp_R':Qlearners2Exp_R, 'TheoreticalAgents_L':TheoreticalAgents_L, 'TheoreticalAgents_R':TheoreticalAgents_R, 'WolfAgents_L':WolfAgents_L, 'WolfAgents_R':WolfAgents_R, 'Random_L':Random_L, 'Random_R':Random_R}

list_player_left = list(map(int,parameters['player_left'].split(', ')))
list_player_right = list(map(int,parameters['player_right'].split(', ')))
algo_left = list(map(str,parameters['algo_left'].split(', ')))
algo_right = list(map(str,parameters['algo_right'].split(', ')))
player_left = []
player_right = []
for i, plNumber in enumerate(list_player_left):
    player_left += possible_algo[algo_left[i]]
    player_left[i].update({'count':plNumber})

for i, plNumber in enumerate(list_player_right):
    player_right += possible_algo[algo_right[i]]
    player_right[i].update({'count':plNumber})


strategies = list(set([bunch['type']+ '_'.join([str(bunch['params'][p]) for p in sorted(bunch['params'].keys())]) for bunch in player_left + player_right]))


output_name = 'output/output_' + '5sims_'
old_method = 2
if old_method == 1:
    output_name += '.method_'
if old_method == 2:
    output_name += 'new.method_'
output_name += 'Rep.' + str(simulation_rounds) + '_'
output_name += 'Time.' + str(simulation_time) + '_'
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
subjective = int(parameters['subjective'])
output_name += 'Subj.' + str(subjective) + '_'
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

windowing = simulation_rounds

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
best_so_far = lower_bound
secretary_calibration_time = simulation_time * 0.3679
andria_calibration_time = simulation_time * 0.3679

# display
verbose_setup = int(parameters['verbose_setup'])
verbose_simulation = int(parameters['verbose_simulation'])
verbose_analysis = int(parameters['verbose_analysis'])
verbose_analysis_matched = int(parameters['verbose_analysis_matched'])
verbose_analysis_non_matched = int(parameters['verbose_analysis_non_matched'])
show_validation = int(parameters['show_validation'])
verbose_silentSimulations = int(parameters['verbose_silentSimulations'])