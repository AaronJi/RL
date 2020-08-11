#!/usr/bin/python
# -*- coding: <encoding name> -*-
import numpy as np
import torch
from python.RLutils.agent.GymAgent import GymAgent

class PongAgent(GymAgent):
    def __init__(self, hyperparams, action_space):
        super(PongAgent, self).__init__(hyperparams, action_space)
        return

    def play(self, state, net, epsilon=0.0, device="cpu"):
        if epsilon is not None and np.random.random() < epsilon:
            action = self.action_space.sample()
        else:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        return action
