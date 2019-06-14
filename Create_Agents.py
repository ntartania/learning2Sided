import numpy as np
from Agent import Agent, ModelBasedAgent, QAgent, QAgent2, Theoragent, WolfAgent, Random
import config
#import Strategy
import Calculation
import math
import random
import numpy as np
import copy




def make_agent(id, agent_type, params):
    if agent_type == 'Model':
        new_agent = ModelBasedAgent(id, agent_type)
    elif agent_type == 'Qlearn':
        new_agent = QAgent(id, agent_type)
    elif agent_type == 'Qlearn2':
        new_agent = QAgent2(id, agent_type)
    elif agent_type == 'Theoretical':
        new_agent = Theoragent(id, agent_type)
    elif agent_type == 'Wolf':
        new_agent = WolfAgent(id, agent_type)
    elif agent_type == 'Random':
        new_agent = Random(id, agent_type)

    else:
        raise NotImplementedError

    new_agent.init_parameters(params)

    return new_agent



def gen_instancelist():
    ID = 0
    ID_to_Agent = {None:None}
    lefties = []
    righties = []
    Left_ID = []
    Right_ID = []
    Left_Attractiveness = []
    Right_Attractiveness = []

    for item in config.player_left:
        # if we are trying with 3 and 17 on the same side, then fix the id
        if item['count'] == 3:
            for z in [4,10,17]:
                new_agent = make_agent(z, item['type'], item['params'])
                lefties.append(new_agent)
                ID_to_Agent[z] = new_agent
                Left_Attractiveness.append(new_agent.attractiveness)
                Left_ID.append(z)
                # print(new_agent.algo_type, new_agent.get_type()[0])
        elif item['count'] == 17:
            for z in [1,2,3,5,6,7,8,9,11,12,13,14,15,16,18,19,20]:
                new_agent = make_agent(z, item['type'], item['params'])
                lefties.append(new_agent)
                ID_to_Agent[z] = new_agent
                Left_Attractiveness.append(new_agent.attractiveness)
                Left_ID.append(z)
                # print(new_agent.algo_type, new_agent.get_type()[0])

        # otherwise assign it normally
        else:
            for cnt in range(item['count']):
                new_agent = make_agent(ID+cnt+1, item['type'], item['params'])
                lefties.append(new_agent)
                ID_to_Agent[ID+cnt+1] = new_agent
                Left_Attractiveness.append(new_agent.attractiveness)
                Left_ID.append(ID+cnt+1)
            ID += item['count']

    for item in config.player_right:
        for cnt in range(item['count']):
            new_agent = make_agent(ID+cnt+1, item['type'], item['params'])
            righties.append(new_agent)
            ID_to_Agent[ID+cnt+1] = new_agent
            Right_Attractiveness.append(new_agent.attractiveness)
            Right_ID.append(ID+cnt+1)
        ID += item['count']

    # Rank Attractiveness
    Left_Ranked_Attractiveness = sorted(Left_Attractiveness)
    Right_Ranked_Attractiveness = sorted(Right_Attractiveness)

    # initialise variables regarding Pearson's correlation coefficient
    Pearson_vector_xy = []
    # print instancelist
    if config.verbose_setup:
        if not config.verbose_silentSimulations: print("player_left cnt=", len(lefties))
        for instance in lefties:
            instance.print_instance()
            # store data for Pearson's correlation coefficient calculation
            for opponent in righties:
                Pearson_vector_xy.append((opponent.attractiveness, instance.get_reward(opponent.get_type())))

        if not config.verbose_silentSimulations: print("player_right cnt=", len(righties))
        for instance in righties:
            instance.print_instance()
            # store data for Pearson's correlation coefficient calculation
            for opponent in lefties:
                Pearson_vector_xy.append((opponent.attractiveness, instance.get_reward(opponent.get_type())))

        # Pearson's correlation coefficient calculation
        Pearson_vector_x, Pearson_vector_y = list(zip(*Pearson_vector_xy))
        if not config.verbose_silentSimulations: print("Pearson's correlation coefficient: {0}".format(Calculation.pearson_correlation_coefficient(Pearson_vector_x, Pearson_vector_y)))

    return lefties, righties, Left_Ranked_Attractiveness, Right_Ranked_Attractiveness