import os
import numpy as np
import visdom
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import MultiStepMemory, PrioritizedMemory

from model import TwinnedQNetwork, GaussianPolicy
import random

date = 'sac-seed0-20210506-1221'
critic = TwinnedQNetwork(2,2,2,[256,256])
critic.load('./logs/dst_d-v0/'+date+'/model/critic.pth')

policy = GaussianPolicy(4,2,[256,256])
policy.load('./logs/dst_d-v0/'+date+'/model/policy.pth')

device = 'cuda'

vis = visdom.Visdom()


def q_heatmap(action,prefer):
    prefer = torch.tensor( prefer,dtype=torch.float32  )
    action = torch.tensor( action,dtype=torch.float32  )

    value = np.empty([11,11])
    time = np.empty([11,11])

    for i in range(11):
        for j in range(11):
            state = torch.tensor( np.array([[i,j]]),dtype=torch.float32 )

            c = critic(state,action,prefer)[0][0].detach().numpy()
            value[10-i][j] = c[0]
            time[10-i][j] = c[1]

    vis.heatmap(X=value,)
    vis.heatmap(X=time,)

def quiver(prefer):
    prefer = torch.tensor( prefer,dtype=torch.float32  )
    x_dir = np.empty([11,11])
    y_dir = np.empty([11,11])
    for i in range(11):
        for j in range(11):
            state = torch.tensor( np.array([[i,j]]),dtype=torch.float32 )
            
            _, _, a = policy.sample(state, prefer)
            a = a.detach().numpy()
            print(a)
            x_dir[10-i][j] = a[0][1]
            y_dir[10-i][j] = -a[0][0]
    vis.quiver(X=x_dir,Y=y_dir)

PREF=[ [0.9, 0.1],[0.8 ,0.2],[0.7 ,0.3],[0.6 ,0.4],[0.5 ,0.5],[0.4 ,0.6],[0.3 ,0.7],[0.2 ,0.8],[0.1 ,0.9] ]

for i in PREF:
    pref = np.array([i])
    quiver(pref)
'''
q_heatmap( np.array( [[1,1]] ),pref )
q_heatmap( np.array( [[1,0]] ),pref )
q_heatmap( np.array( [[1,-1]] ),pref )
q_heatmap( np.array( [[0,-1]] ),pref )
q_heatmap( np.array( [[-1,-1]] ),pref )
q_heatmap( np.array( [[-1,0]] ),pref )
q_heatmap( np.array( [[-1,1]] ),pref )
q_heatmap( np.array( [[0,1]] ),pref )
'''


