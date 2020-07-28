#!/usr/bin/python
# -*- coding: <encoding name> -*-

from python.RLutils.agent.GymAgent import GymAgent

class PongAgent(GymAgent):
    def __init__(self, hyperparams, env):
        super(PongAgent, self).__init__(hyperparams, env)
        #self.exp_buffer = exp_buffer

