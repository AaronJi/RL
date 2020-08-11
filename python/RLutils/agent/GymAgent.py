#!/usr/bin/python
# -*- coding: <encoding name> -*-

from python.RLutils.agent.Agent import Agent

class GymAgent(Agent):
    def __init__(self, hyperparams, action_space):
        super(GymAgent, self).__init__(hyperparams)
        self.action_space = action_space
        return
