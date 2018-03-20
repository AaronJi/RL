#!/usr/bin/python
# -*- coding: <encoding name> -*-

import os
import sys
import numpy as np

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from RLutils.agent.Agent import Agent
from RLutils.algorithm.utils import softmax


class MDPrankAgent(Agent):

    def __init__(self, hyperparams, theta0):
        super(MDPrankAgent, self).__init__(hyperparams)
        self.theta = theta0
        self.nParam = len(theta0)
        return

    # perform an action according to the policy, given the current state
    def act(self, state, random=True):
        assert self._hyperparams["policyTYpe"] == "stochastic"
        # if the policy is stochastic, this is a MonteCarlo sampling

        actions = self.getActionList(state)

        pi = self.calPolicyProbMap(state)

        assert np.abs(np.sum(pi) - 1.0) < 1.0e-5

        if random:
            randNum = np.random.rand(1)  # a random number between [0, 1] according to the uniform distribution

            for action, action_prob in zip(actions, pi):
                randNum -= action_prob
                if randNum < 0:
                    return action

            print("warning: have not selected an action; should NOT happen")
        else:
            ia = np.argmax(pi)
            return actions[ia]

        return actions[-1]


    # list of all possible actions given the current state
    def getActionList(self, state):
        candidates = state[1]
        return range(len(candidates))

    # the probability of action given a state
    def calPolicyProbMap(self, state):
        assert self._hyperparams["policyTYpe"] == "stochastic"

        actions = self.getActionList(state)

        hvals = np.array([self.h(state, action) for action in actions])

        # pi(a|s), the probability of executed action given the current state
        pi = softmax(hvals)

        self.pi = pi

        return pi

    # an evaluation function of state and action; current just use a linear model
    def h(self, state, action):
        candidates = state[1]

        # the list of candidates which are not ranked yet; each candidate contains its feature and label
        candidate = candidates[action]
        x = candidate[0]

        assert self.nParam == x.shape[0]
        h = np.dot(self.theta, x)

        return h

