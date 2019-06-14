import numpy as np
import Create_Agents
import random
import config
import Agent
import matplotlib.pyplot as plt
import dill
from scipy import stats

files = ['Left 20 modelers_L | Right 20 modelers_R | Rounds 70 | Steps 100 | BASELINE COUNT vs BASELINE COUNT.pkl',
         'Left 20 modelers_L | Right 20 modelers_R | Rounds 70 | Steps 100 | EPSILON_THRESHOLD vs BASELINE COUNT.pkl',
         'Left 20 modelers_L | Right 20 modelers_R | Rounds 70 | Steps 100 | EPSILON_THRESHOLD vs EPSILON_THRESHOLD.pkl',
         'Left 20 modelers_L | Right 20 modelers_R | Rounds 70 | Steps 100 | KNN vs BASELINE COUNT.pkl',
         'Left 20 modelers_L | Right 20 modelers_R | Rounds 70 | Steps 100 | KNN vs EPSILON_THRESHOLD.pkl',
         'Left 20 modelers_L | Right 20 modelers_R | Rounds 70 | Steps 100 | KNN vs KNN.pkl',
         'Left 20 modelers_L | Right 20 Qlearners2_R | Rounds 70 | Steps 100 | BASELINE COUNT vs QLearning2.pkl',
         'Left 20 modelers_L | Right 20 Qlearners2_R | Rounds 70 | Steps 100 | EPSILON_THRESHOLD vs QLearning2.pkl',
         'Left 20 modelers_L | Right 20 Qlearners2_R | Rounds 70 | Steps 100 | KNN vs QLearning2.pkl',
         'Left 20 modelers_L | Right 20 Qlearners2Exp_R | Rounds 70 | Steps 100 | BASELINE COUNT vs QLearning2 EXPLORATION.pkl',
         'Left 20 modelers_L | Right 20 Qlearners2Exp_R | Rounds 70 | Steps 100 | EPSILON_THRESHOLD vs QLearning2 EXPLORATION.pkl',
         'Left 20 modelers_L | Right 20 Qlearners2Exp_R | Rounds 70 | Steps 100 | KNN vs QLearning2 EXPLORATION.pkl',
         'Left 20 Qlearners2_L | Right 20 Qlearners2_R | Rounds 70 | Steps 100 | QLearning2 vs QLearning2.pkl',
         'Left 20 Qlearners2Exp_L | Right 20 Qlearners2_R | Rounds 70 | Steps 100 | QLearning2 EXPLORATION vs QLearning2.pkl',
         'Left 20 Qlearners2Exp_L | Right 20 Qlearners2Exp_R | Rounds 70 | Steps 100 | QLearning2 EXPLORATION vs QLearning2 EXPLORATION.pkl']

matrixFile = open('/Users/Riccardo/Desktop/matrixVals.txt', 'w')
for name in files:
    with open('output/' + name, 'rb') as f:
        agents_to_plot_RIGHT, agents_to_plot_LEFT, playerinfo, playerlearning = dill.load(f)
        x_RIGHT = np.array([])
        y_RIGHT = np.array([])
        x_LEFT = np.array([])
        y_LEFT = np.array([])

        for i in agents_to_plot_RIGHT:
            i.utility = [x for x in i.utility if x != 0]
            x_R_temp = np.full((len(i.utility)), i.get_type()[0])
            x_RIGHT = np.concatenate((x_RIGHT, x_R_temp))
            y_RIGHT = np.concatenate((y_RIGHT, i.utility))

        for i in agents_to_plot_LEFT:
            i.utility = [x for x in i.utility if x != 0]
            x_L_temp = np.full((len(i.utility)), i.get_type()[0])
            x_LEFT = np.concatenate((x_LEFT, x_L_temp))
            y_LEFT = np.concatenate((y_LEFT, i.utility))

        slopeL, intercept, r_value, p_value, std_err = stats.linregress(x_LEFT, y_LEFT)
        matrixFile.write(name+" [Left] : "+str(slopeL)+"\n")

        slopeR, intercept, r_value, p_value, std_err = stats.linregress(x_RIGHT, y_RIGHT)
        matrixFile.write(name+" [Right] : "+str(slopeR)+"\n________________________________\n\n")

matrixFile.close()
