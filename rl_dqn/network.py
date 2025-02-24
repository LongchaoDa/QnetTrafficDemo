# network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        """
        x: shape [batch_size, state_dim]
        return: shape [batch_size, action_dim], 每个action的Q值
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
