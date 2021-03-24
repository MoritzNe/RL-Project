# Imports
import random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from collections import OrderedDict
#from new_hope_alien import DQN

class DQN(nn.Module):
    """
    DQN class, whose q-values will be visualized
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        #self.fc3 = nn.Linear(in_features=4, out_features=4)
        #self.fc4 = nn.Linear(in_features=2, out_features=2)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, input):
        input = F.rrelu(self.fc1(input))
        input = F.rrelu(self.fc2(input))
        #input = F.rrelu(self.fc3(input))
        #input = F.leaky_relu(self.fc4(input))
        output = self.out(input)
        return output


def main():

    net = DQN()
    #net.load_state_dict(state_2l_16_32_1000iter)
    net.load_state_dict(torch.load('policy_net'))
    net.eval()
    list_of_rew_pos = [-0.5, -0.25, 0.0, 0.25, 0.5]
    fig, ax = plt.subplots()
    greys = ['lightgrey', 'darkgrey', 'grey', 'dimgrey', 'black']
    purples = ['mistyrose', 'pink', 'orchid', 'darkviolet', 'navy']

    for i, reward_pos in enumerate(list_of_rew_pos):

            arr = np.empty(400, dtype=np.float32)
            arr.fill(reward_pos)
            input = torch.stack((torch.from_numpy(np.linspace(-0.5, 0.5, 400, dtype=np.float32)), torch.from_numpy(arr)), 1)
            output = net(input)

            line1, = ax.plot(output[:, 0].detach().numpy(), label=f'L, {int((reward_pos + 0.5) * 400)}', color=greys[i])
            line2, = ax.plot(output[:, 1].detach().numpy(), label=f'R, {int((reward_pos + 0.5) * 400)}', color=purples[i])

    plt.legend(ncol=5, loc='upper center', fontsize=8)
    plt.xlabel('position of player in pixels')
    plt.ylabel('qvalue')
    plt.title('qvalue of the action for player given reward at position')

    #plt.show()
    fig.savefig("qfuncs.pdf")