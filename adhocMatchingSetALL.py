import numpy as np
#import Agent
import matplotlib.pyplot as plt
#import Plot
import time
#import ExpectedValue
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dill
import glob
import os

cd = os.getcwd()
path = "./output/Pickles/"
os.chdir(path)
left = "Wolf"
right = "Wolf"
gamma = 0.9
agents_per_side = 20
# left = "Qlearning Exp"
# right = left
pickleFile = "Wolf vs Wolf | r20s400.tsv"

agents_LEFT = {}
agents_RIGHT = {}

def mean(l):
    return sum(l)*1.0/len(l)

def expU(match_list, num_agents,thegamma):
    expV = mean(match_list)
    p = len(match_list) *1.0 / num_agents
    if p != 1:
        EU = ( p * (1-p) * expV ) / ( 1 - thegamma + (thegamma*p) )
    else:
        EU = ( p * expV ) / ( 1 - thegamma + (thegamma*p) )
    return EU

expUtility_LEFT = {}
expUtility_RIGHT = {}
roundcount=0
with open(pickleFile, 'r') as f:
    for line in f:
        roundcount+=1
        pairs = [tuple(p.split(',')) for p in line.split('\t')]
        for l,r in pairs:
            agents_LEFT.setdefault(int(l),[]).append(int(r))
            agents_RIGHT.setdefault(int(r),[]).append(int(l))

        for k in range(1,21):
            if (k in agents_LEFT):
                expUtility_LEFT[k]=expUtility_LEFT.get(k,0)+ expU(agents_LEFT[k], agents_per_side, gamma) 
            #else:
            #    expUtility_LEFT[k]=expUtility_LEFT.get(k,0)
            if (k in agents_RIGHT):
                expUtility_RIGHT[k]=expUtility_RIGHT.get(k,0)+expU(agents_RIGHT[k], agents_per_side, gamma)
            #else:
            #    expUtility_RIGHT[k]=0

        allwouldmatches = []
        for kl in agents_LEFT: #I could use the right or the left perspective it's just inverted but the info is the same
            allwouldmatches += [(kl,match) for match in agents_LEFT[kl]]

for k in range(1,21):
    expUtility_LEFT[k]=expUtility_LEFT.get(k,0)*1.0/ roundcount 
    expUtility_RIGHT[k]=expUtility_RIGHT.get(k,0)*1.0/ roundcount 
    print 'ExpU[',k,']:',expUtility_LEFT[k],expUtility_RIGHT[k]


fig, ax= plt.subplots()
# print(match_LEFT)
x = [t[0] for t in allwouldmatches]
y = [t[1] for t in allwouldmatches]

ax.plot(y, x, 'o', c='b')
ax.set_xlabel("Right Agent Type", fontsize=15)
ax.set_ylabel("Left Agent Type", fontsize=15)
ax.set_title(left+" vs "+right+ r'   $\gamma$:'+str(gamma), fontsize=19)
ax.set_xticks(np.arange(0, 21, 1.0))
ax.set_yticks(np.arange(0, 21, 1.0))

fig.savefig("../Matching_"+left+"_"+right+"_"+str(gamma)+".pdf")
plt.close(fig)