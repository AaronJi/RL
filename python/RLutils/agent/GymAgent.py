#!/usr/bin/python
# -*- coding: <encoding name> -*-

import numpy as np
import torch

from python.RLutils.algorithm.experience_buffer import Experience
from python.RLutils.agent.Agent import Agent

class GymAgent(Agent):
    def __init__(self, hyperparams, env):
        super(GymAgent, self).__init__(hyperparams)
        self.env = env
        #self._reset()

    #def _reset(self):
    #    self.state = self.env.reset()
    #    self.total_reward = 0.0


    def play(self, state, net, epsilon):
        if np.random.random() < epsilon:
            action = self.env.get_action_space().sample()
        else:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a).to(device='cpu')
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        return action

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.get_action_space().sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward
